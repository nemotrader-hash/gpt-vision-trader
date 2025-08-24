#!/usr/bin/env python3
import asyncio
import base64
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str = (
        "Analyse ce graphique financier. "
        "Décris en quelques phrases ce que tu vois (indicateurs, tendances, motifs). "
        "Ensuite, d'après ton analyse des motifs visuels, prédis si le mouvement des prix "
        "sur les 30 prochaines chandelles sera bullish, bearish, ou neutral. "
        "Tu DOIS répondre avec un objet JSON contenant deux clés : 'analysis' pour ton analyse textuelle, "
        "et 'prediction' pour ta prédiction (bullish, bearish, ou neutral). "
        'Par exemple: {"analysis": "L\'analyse du graphique montre...", "prediction": "bullish"}.'
    )
    prediction_key: str = "prediction"
    analysis_key: str = "analysis"
    valid_predictions: frozenset[str] = frozenset({"bullish", "bearish", "neutral"})


class AnalysisService(ABC):
    """Abstract base class defining the interface for image analysis services."""
    
    def __init__(self, prompt: Prompt):
        self._prompt = prompt
    
    @abstractmethod
    async def analyze_image(self, image_path: str, image_b64: str) -> str:
        """
        Analyze an image and return a JSON string response.
        
        Args:
            image_path: Path to the image file (for reference/logging)
            image_b64: Base64 encoded image data
            
        Returns:
            JSON string containing prediction and analysis
        """
        pass
    
    @abstractmethod
    async def __aenter__(self):
        """Initialize resources when entering async context."""
        pass
        
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources when exiting async context."""
        pass
    
    async def finalize(self) -> None:
        """
        Finalize any pending operations and cleanup.
        Must be called after all analyze_image calls are done.
        Deprecated: Use context manager instead.
        """
        pass


class GPTAnalysisService(AnalysisService):
    
    CALLS_PER_MINUTE = 50
    SLEEP_TIME = 60 / CALLS_PER_MINUTE 
    
    def __init__(self, prompt: Prompt, api_key: str, gpt_model: str = "gpt-4-vision-preview"):
        super().__init__(prompt)
        self._api_key = api_key
        self._client = None
        self._last_call_time: float = 0.0
        self._gpt_model = gpt_model
    
    async def __aenter__(self):
        """Initialize OpenAI client when entering async context."""
        self._client = self._init_client(self._api_key)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """No special cleanup needed."""
        pass
    
    def _init_client(self, api_key: str) -> OpenAI:
        if not api_key:
            logging.critical("The OpenAI API key must be provided.")
            raise EnvironmentError("Missing OpenAI API key")
        try:
            client = OpenAI(api_key=api_key)
            logging.info("OpenAI client initialized successfully.")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def analyze_image(self, image_path: str, image_b64: str) -> str:
        if not self._client:
            raise RuntimeError("Service must be used within an async context manager")
            
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self.SLEEP_TIME:
            sleep_duration = self.SLEEP_TIME - time_since_last_call
            await asyncio.sleep(sleep_duration)
        
        response = self._client.chat.completions.create(
            model=self._gpt_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._prompt.text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        self._last_call_time = time.time()
        if not response.choices[0].message.content:
            raise ValueError("Empty response from GPT API")
        return response.choices[0].message.content


class BatchGPTAnalysisService(AnalysisService):
    """Service that uses OpenAI's Batch API for efficient bulk image analysis."""
    
    BATCH_SIZE = 60
    CHECK_DELAY: float = 5
    BACKOFF_FACTOR: float = 1.1
    
    def __init__(self, prompt: Prompt, api_key: str, gpt_model: str = "gpt-4-vision-preview"):
        super().__init__(prompt)
        self._api_key = api_key
        self._client: OpenAI | None = None
        self._pending_tasks: list[dict] = []
        self._batch_results: dict[str, str] = {}
        self._processing_complete = False
        self._active_batch_ids: list[tuple[str, str]] = [] 
        self._gpt_model = gpt_model
    
    async def __aenter__(self):
        self._client = self._init_client(self._api_key)
        self._processing_complete = False
        self._active_batch_ids = []
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Process all pending tasks in batches concurrently."""
        if not self._pending_tasks:
            self._processing_complete = True
            return
            
        batches = [
            self._pending_tasks[i:i + self.BATCH_SIZE]
            for i in range(0, len(self._pending_tasks), self.BATCH_SIZE)
        ]
        
        submit_futures = [self._submit_batch(batch) for batch in batches]
        await asyncio.gather(*submit_futures)
        
        process_futures = [self._process_batch_results(batch_id, file_id) 
                         for batch_id, file_id in self._active_batch_ids]
        await asyncio.gather(*process_futures)
            
        self._pending_tasks = []
        self._active_batch_ids = []
        self._processing_complete = True

    def _init_client(self, api_key: str) -> OpenAI:
        if not api_key:
            logging.critical("The OpenAI API key must be provided.")
            raise EnvironmentError("Missing OpenAI API key")
        try:
            client = OpenAI(api_key=api_key)
            logging.info("OpenAI client initialized successfully.")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def analyze_image(self, image_path: str, image_b64: str) -> str:
        """Queue image for batch analysis."""
        if not self._client:
            raise RuntimeError("Service must be used within an async context manager")
        
        task = {
            "custom_id": image_path,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self._gpt_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt.text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "max_tokens": 500,
                "response_format": {"type": "json_object"},
            }
        }
        self._pending_tasks.append(task)
        return ""

    async def _submit_batch(self, tasks: list[dict]) -> None:
        """Submit a batch of tasks WITHOUT waiting for completion."""
        if not self._client or not tasks:
            return
            
        batch_file_path = f"batch_tasks_{int(time.time())}.jsonl"
        try:
            with open(batch_file_path, "w") as f:
                for task in tasks:
                    f.write(json.dumps(task) + "\n")
            
            with open(batch_file_path, "rb") as f:
                batch_file = self._client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            batch_job = self._client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Store batch ID and file ID for later processing
            self._active_batch_ids.append((batch_job.id, batch_file.id))
            
        finally:
            try:
                os.remove(batch_file_path)
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file {batch_file_path}: {e}")

    async def _process_batch_results(self, batch_id: str, file_id: str) -> None:
        """Wait for a batch to complete and process its results."""
        if not self._client:
            return
            
        current_delay = self.CHECK_DELAY
        while True:
            status = self._client.batches.retrieve(batch_id)
            logging.info(f"Batch {batch_id} status: {status.status} (waiting {current_delay:.1f}s before next check)")
            
            if status.status == "completed":
                if not status.output_file_id:
                    raise RuntimeError(f"Batch {batch_id} completed but no output file ID")
                    
                result_file = self._client.files.content(status.output_file_id)
                content = result_file.content.decode()
                logging.info(f"Processing batch results: {content}")
                
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line)
                        if "custom_id" not in result or "response" not in result:
                            logging.error(f"Invalid result format: {line}")
                            continue
                            
                        image_path = result["custom_id"]
                        if "body" not in result["response"] or "choices" not in result["response"]["body"]:
                            logging.error(f"Invalid response format for {image_path}: {result}")
                            continue
                            
                        choices = result["response"]["body"]["choices"]
                        if not choices or "message" not in choices[0]:
                            logging.error(f"No choices or message for {image_path}")
                            continue
                            
                        content = choices[0]["message"]["content"]
                        self._batch_results[image_path] = content
                        logging.info(f"Processed result for {image_path}: {content}")
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON in result: {line}")
                    except Exception as e:
                        logging.error(f"Error processing result: {e}")
                break
                
            elif status.status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed")
                
            await asyncio.sleep(current_delay)
            current_delay *= self.BACKOFF_FACTOR

    def get_cached_results(self) -> dict[str, str]:
        """Return all cached results only if processing is complete."""
        if not self._processing_complete:
            raise RuntimeError("Cannot get results before all batches have completed")
        return self._batch_results.copy()


class MockAnalysisService(AnalysisService):
    """Mock service that returns random predictions for testing."""
    
    async def __aenter__(self):
        """No special initialization needed."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """No special cleanup needed."""
        pass
    
    async def analyze_image(self, image_path: str, image_b64: str) -> str:
        prediction = random.choice(list(self._prompt.valid_predictions))
        mock_analysis = f"This is a mock analysis for {os.path.basename(image_path)}. The prediction is {prediction}."
        
        mock_response = {
            self._prompt.prediction_key: prediction,
            self._prompt.analysis_key: mock_analysis
        }
        
        return json.dumps(mock_response)


class ImageAnalyzer:
    """A class to analyze financial chart images using a configurable analysis service."""

    def __init__(self, analysis_service: AnalysisService):
        self._service = analysis_service
        self._prompt = analysis_service._prompt
        self._valid_predictions = self._prompt.valid_predictions
        self._client_initialized = False
    
    @property
    def supports_batch(self) -> bool:
        """Whether the analyzer supports batch processing."""
        return isinstance(self._service, BatchGPTAnalysisService)
    
    async def _ensure_client_initialized(self):
        """Initialize client once and only once."""
        if not self._client_initialized:
            await self._service.__aenter__()
            self._client_initialized = True
    
    async def __aenter__(self):
        await self._ensure_client_initialized()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client_initialized:
            await self._service.__aexit__(exc_type, exc_val, exc_tb)
            self._client_initialized = False

    async def analyze(self, image_path: str) -> tuple[str, str | None]:
        """
        Analyze a single financial chart image and return prediction and analysis.
        
        Args:
            image_path: Path to the image file to analyze

        Returns:
            tuple[str, str | None]: (prediction, analysis_text)
            prediction is one of the valid predictions or an error string
            analysis_text is the textual analysis or None if an error occurred
        """
        if self.supports_batch:
            raise TypeError("Use analyze_batch() for batch-capable services")
            
        if not self._validate_image_path(image_path):
            return "error_image_not_found", None

        try:
            image_b64 = self._encode_image_to_base64(image_path)
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {e}")
            return "error_encoding_image", None

        try:
            await self._ensure_client_initialized()
            response = await self._service.analyze_image(image_path, image_b64)
            return self._process_response(response, image_path)
        except Exception as e:
            logging.error(f"Analysis service call failed for {image_path}: {e}")
            return "error_analysis_service_call", None

    async def analyze_batch(self, image_paths: list[str]) -> list[tuple[str, str | None]]:
        """
        Analyze multiple images in batch mode.
        
        Args:
            image_paths: List of paths to image files to analyze
            
        Returns:
            List of (prediction, analysis_text) tuples
        """
        if not self.supports_batch:
            raise TypeError("Batch analysis not supported by current service")
            
        if not isinstance(self._service, BatchGPTAnalysisService):
            raise TypeError("Service must be BatchGPTAnalysisService for batch operations")
            
        results: list[tuple[str, str | None]] = []
        error_indices: dict[int, tuple[str, str | None]] = {}
        
        try:
            await self._ensure_client_initialized()
            
            for i, image_path in enumerate(image_paths):
                if not self._validate_image_path(image_path):
                    error_indices[i] = ("error_image_not_found", None)
                    continue

                try:
                    image_b64 = self._encode_image_to_base64(image_path)
                    await self._service.analyze_image(image_path, image_b64)
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")
                    error_indices[i] = ("error_processing_image", None)
                    continue
            
            # Initialize results list with placeholders
            results = [("pending", None)] * len(image_paths)
            
            # Fill in known errors
            for idx, error_result in error_indices.items():
                results[idx] = error_result
            
            # Second phase: Wait for all batches to complete
            await self._service.__aexit__(None, None, None)
            self._client_initialized = False
            
            # Third phase: Get results only after all processing is done
            batch_results = self._service.get_cached_results()  # Now type-safe
            
            # Fill in successful results while preserving order
            for i, image_path in enumerate(image_paths):
                if i in error_indices:
                    continue
                    
                if image_path in batch_results:
                    results[i] = self._process_response(batch_results[image_path], image_path)
                else:
                    logging.error(f"No result found for {image_path} after batch completion")
                    results[i] = ("error_batch_processing", None)
                    
        except Exception as e:
            logging.error(f"Error in batch analysis: {e}")
            if self._client_initialized:
                await self._service.__aexit__(None, None, None)
                self._client_initialized = False
            raise
            
        return results

    def _validate_image_path(self, image_path: str) -> bool:
        """Validate that the image file exists."""
        if not os.path.isfile(image_path):
            logging.warning(f"Image file not found: {image_path}")
            return False
        return True

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Read an image file and return its base64-encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _process_response(self, raw_response: str, image_path: str) -> tuple[str, str | None]:
        """Extract and validate prediction and analysis from parsed response."""
        try:
            parsed_response = json.loads(raw_response)

            if not isinstance(parsed_response, dict):
                logging.error(
                    f"Parsed response is not a dictionary for {image_path}. Response: {raw_response}"
                )
                return "error_json_not_dict", None

            analysis_text = parsed_response.get(self._prompt.analysis_key)
            prediction = parsed_response.get(self._prompt.prediction_key, "").lower()

            if not analysis_text:
                logging.warning(
                    f"Analysis text missing in JSON response for {image_path}. Full JSON: {raw_response}"
                )
                analysis_text = "No analysis provided by model."
            else:
                logging.info(f"Analysis for {image_path}: {analysis_text}")

            logging.info(f"Prediction for {image_path}: {prediction}")

            if prediction not in self._valid_predictions:
                logging.warning(
                    f"Unexpected prediction value '{prediction}' in JSON for {image_path}. Full JSON: {raw_response}"
                )
                return "error_invalid_prediction_value", analysis_text

            return prediction, analysis_text

        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to decode JSON response for {image_path}: {e}. Response: {raw_response}"
            )
            return "error_json_decode", None
        except Exception as e:
            logging.error(
                f"Error processing JSON response for {image_path}: {e}. Response: {raw_response}"
            )
            return "error_processing_json", None


class MetadataProcessor:
    """Process and update metadata JSON files with image analysis results."""

    def __init__(self, analyzer: ImageAnalyzer, metadata_path: str, images_folder: str):
        """
        Initialize the processor with required configuration.

        Args:
            analyzer: ImageAnalyzer instance to use for processing images
            metadata_path: Path to the metadata JSON file
            images_folder: Path to the folder containing images
        """
        self._analyzer = analyzer
        self._metadata_path = metadata_path
        self._images_folder = images_folder

    async def process(self) -> None:
        """Process all images and update the metadata JSON file."""
        metadata = self._load_metadata()
        if metadata is None:
            return

        try:
            image_paths = [
                os.path.join(self._images_folder, filename)
                for filename in metadata.keys()
            ]
            
            # Initialize client once for all operations
            async with self._analyzer as analyzer:
                if analyzer.supports_batch:
                    responses = await analyzer.analyze_batch(image_paths)
                    for filename, (prediction, analysis) in zip(metadata.keys(), responses):
                        metadata[filename] = self._update_metadata(
                            metadata[filename], 
                            prediction,
                            analysis
                        )
                else:
                    for filename, data in metadata.items():
                        logging.info(f"Processing image: {filename}")
                        image_path = os.path.join(self._images_folder, filename)
                        prediction, analysis = await analyzer.analyze(image_path)
                        metadata[filename] = self._update_metadata(data, prediction, analysis)
            
            self._save_metadata(metadata)
            
        except Exception as e:
            logging.error(f"Error during processing: {e}")

    def _update_metadata(self, current_data: dict, prediction: str, analysis: str | None) -> dict:
        """Update metadata for a single image with analysis results."""
        metadata = current_data if isinstance(current_data, dict) else {"legacy_value": current_data}
        metadata["prediction_gpt"] = prediction
        metadata["analysis_gpt"] = analysis if analysis is not None else "Analysis not available"
        return metadata

    def _load_metadata(self) -> dict | None:
        """
        Load JSON metadata with UTF-8 encoding, falling back to latin-1 if needed.
        Returns None if loading fails.
        """
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logging.info(f"Successfully read metadata file {self._metadata_path} with UTF-8 encoding.")
            return metadata
        except UnicodeDecodeError:
            logging.warning(f"Failed to read {self._metadata_path} with UTF-8 encoding. Attempting with latin-1.")
            try:
                with open(self._metadata_path, "r", encoding="latin-1") as f:
                    metadata = json.load(f)
                logging.info(f"Successfully read metadata file {self._metadata_path} with latin-1 encoding.")
                return metadata
            except Exception as e:
                logging.error(f"Failed to read metadata file with latin-1 encoding: {e}")
                return None
        except FileNotFoundError:
            logging.error(f"Metadata file not found: {self._metadata_path}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in file: {self._metadata_path}. Error: {e}")
            return None
        except Exception as e:
            logging.error(f"Error reading metadata file {self._metadata_path}: {e}")
            return None

    def _save_metadata(self, metadata: dict) -> None:
        """Save updated metadata back to JSON file."""
        try:
            with open(self._metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved metadata to {self._metadata_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata to {self._metadata_path}: {e}")


class PredictionAnalyzer:
    """Analyzes and compares GPT predictions with actual trend data."""

    def __init__(self, metadata_path: str):
        """
        Initialize the analyzer with metadata file path.
        
        Args:
            metadata_path: Path to the metadata JSON file containing both predictions and trends
        """
        self._metadata_path = metadata_path
        self._metadata = self._load_metadata()

    def save_analysis_to_file(self, analysis: dict, output_file: str) -> None:
        """
        Save analysis report to the specified output file.
        
        Args:
            analysis: Dictionary containing analysis results
            output_file: Path where to save the analysis report
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=== Prediction Analysis Report ===\n\n")
                
                f.write(f"Total Samples: {analysis['total_samples']}\n")
                f.write(f"Overall Accuracy: {analysis['accuracy']:.2%}\n\n")
                
                f.write("Distribution of Predictions:\n")
                for pred, count in analysis['gpt_predictions'].items():
                    percentage = count / analysis['total_samples'] * 100
                    f.write(f"  {pred.capitalize()}: {count} ({percentage:.1f}%)\n")
                    
                f.write("\nDistribution of Actual Trends:\n")
                for trend, count in analysis['actual_trends'].items():
                    percentage = count / analysis['total_samples'] * 100
                    f.write(f"  {trend.capitalize()}: {count} ({percentage:.1f}%)\n")
                    
                f.write("\nConfusion Matrix:\n")
                f.write("Actual \\ Predicted | Bullish | Bearish | Neutral\n")
                f.write("-" * 45 + "\n")
                for actual in ["bullish", "bearish", "neutral"]:
                    pred_counts = analysis['confusion_matrix'][actual]
                    f.write(f"{actual.capitalize():15} | {pred_counts['bullish']:7} | {pred_counts['bearish']:7} | {pred_counts['neutral']:7}\n")
                    
                f.write("\nDetailed Metrics Analysis:\n")
                for trend, metrics in analysis['class_metrics'].items():
                    f.write(f"\n{trend.capitalize()}:\n")
                    true_pos = analysis['confusion_matrix'][trend][trend]
                    false_pos = sum(analysis['confusion_matrix'][t][trend] for t in ["bullish", "bearish", "neutral"]) - true_pos
                    false_neg = sum(analysis['confusion_matrix'][trend][p] for p in ["bullish", "bearish", "neutral"]) - true_pos
                    
                    f.write(f"  Precision: {metrics['precision']:.2%}\n")
                    f.write(f"    → Out of {true_pos + false_pos} {trend} predictions, {true_pos} were correct\n")
                    f.write(f"    → When GPT predicts {trend}, it's right {metrics['precision']:.2%} of the time\n")
                    
                    f.write(f"  Recall: {metrics['recall']:.2%}\n")
                    f.write(f"    → Out of {true_pos + false_neg} actual {trend} cases, caught {true_pos}\n")
                    f.write(f"    → GPT catches {metrics['recall']:.2%} of all {trend} trends\n")
                    
                    f.write(f"  F1 Score: {metrics['f1']:.2%}\n")
                    
            logging.info(f"Analysis report saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save analysis report to {output_file}: {e}")
    
    def print_analysis(self, output_file: str | None = None) -> None:
        """
        Print a formatted analysis report and optionally save to file.
        
        Args:
            output_file: Optional path to save the analysis report. If None, only prints to console.
        """
        analysis = self.analyze()
        if not analysis:
            print("No data available for analysis")
            return
            
        print("\n=== Prediction Analysis Report ===\n")
        
        print(f"Total Samples: {analysis['total_samples']}")
        print(f"Overall Accuracy: {analysis['accuracy']:.2%}\n")
        
        print("Distribution of Predictions:")
        for pred, count in analysis['gpt_predictions'].items():
            percentage = count / analysis['total_samples'] * 100
            print(f"  {pred.capitalize()}: {count} ({percentage:.1f}%)")
            
        print("\nDistribution of Actual Trends:")
        for trend, count in analysis['actual_trends'].items():
            percentage = count / analysis['total_samples'] * 100
            print(f"  {trend.capitalize()}: {count} ({percentage:.1f}%)")
            
        print("\nConfusion Matrix:")
        print("Actual \\ Predicted | Bullish | Bearish | Neutral")
        print("-" * 45)
        for actual in ["bullish", "bearish", "neutral"]:
            pred_counts = analysis['confusion_matrix'][actual]
            print(f"{actual.capitalize():15} | {pred_counts['bullish']:7} | {pred_counts['bearish']:7} | {pred_counts['neutral']:7}")
            
        print("\nDetailed Metrics Analysis:")
        for trend, metrics in analysis['class_metrics'].items():
            print(f"\n{trend.capitalize()}:")
            true_pos = analysis['confusion_matrix'][trend][trend]
            false_pos = sum(analysis['confusion_matrix'][t][trend] for t in ["bullish", "bearish", "neutral"]) - true_pos
            false_neg = sum(analysis['confusion_matrix'][trend][p] for p in ["bullish", "bearish", "neutral"]) - true_pos
            
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"    → Out of {true_pos + false_pos} {trend} predictions, {true_pos} were correct")
            print(f"    → When GPT predicts {trend}, it's right {metrics['precision']:.2%} of the time")
            
            print(f"  Recall: {metrics['recall']:.2%}")
            print(f"    → Out of {true_pos + false_neg} actual {trend} cases, caught {true_pos}")
            print(f"    → GPT catches {metrics['recall']:.2%} of all {trend} trends")
            
            print(f"  F1 Score: {metrics['f1']:.2%}")
            
        # Save to file if output_file is specified
        if output_file:
            self.save_analysis_to_file(analysis, output_file)

    def analyze(self) -> dict:
        """
        Perform statistical analysis of predictions vs trends.
        
        Returns:
            Dictionary containing various statistical metrics
        """
        if not self._metadata:
            return {}
            
        total_samples = len(self._metadata)
        matches = 0
        confusion_matrix = {
            "bullish": {"bullish": 0, "bearish": 0, "neutral": 0},
            "bearish": {"bullish": 0, "bearish": 0, "neutral": 0},
            "neutral": {"bullish": 0, "bearish": 0, "neutral": 0}
        }
        
        gpt_predictions = {"bullish": 0, "bearish": 0, "neutral": 0}
        actual_trends = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        for data in self._metadata.values():
            if not isinstance(data, dict):
                continue
                
            gpt_pred = data.get("prediction_gpt", "")
            actual_trend = data.get("trend", "")
            
            if not gpt_pred or not actual_trend:
                continue
                
            gpt_predictions[gpt_pred] += 1
            actual_trends[actual_trend] += 1
            
            confusion_matrix[actual_trend][gpt_pred] += 1
            
            if gpt_pred == actual_trend:
                matches += 1
        
        accuracy = matches / total_samples if total_samples > 0 else 0
        
        metrics = {}
        for trend in ["bullish", "bearish", "neutral"]:
            true_pos = confusion_matrix[trend][trend]
            false_pos = sum(confusion_matrix[t][trend] for t in ["bullish", "bearish", "neutral"]) - true_pos
            false_neg = sum(confusion_matrix[trend][p] for p in ["bullish", "bearish", "neutral"]) - true_pos
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[trend] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        return {
            "total_samples": total_samples,
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix,
            "gpt_predictions": gpt_predictions,
            "actual_trends": actual_trends,
            "class_metrics": metrics
        }
    
    def _load_metadata(self) -> dict | None:
        """Load metadata from JSON file."""
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading metadata for analysis: {e}")
            return None


async def process_images_and_update_json(
    metadata_path: str | list[str], 
    images_folder: str, 
    api_key: str | None = None,
    use_batch: bool = False,
    gpt_model: str = "gpt-4o-mini"
) -> None:
    """
    Process images and update metadata JSON file(s) with GPT analysis results.
    
    Args:
        metadata_path: Path to metadata JSON file or list of paths to multiple metadata files
        images_folder: Path to the folder containing images
        api_key: OpenAI API key. If provided, uses GPT service. If None, uses mock service.
        use_batch: If True and api_key is provided, uses BatchGPTAnalysisService instead of GPTAnalysisService
        gpt_model: The GPT model to use for analysis (default: "gpt-4o-mini")
    """
    prompt = Prompt()
    
    service: Union[BatchGPTAnalysisService, GPTAnalysisService, MockAnalysisService]
    
    match (api_key, use_batch):
        case (str() as key, True): 
            service = BatchGPTAnalysisService(prompt, key, gpt_model)
        case (str() as key, False):  
            service = GPTAnalysisService(prompt, key, gpt_model)
        case (None, _): 
            service = MockAnalysisService(prompt)
    
    # Convert single path to list for uniform processing
    metadata_paths = [metadata_path] if isinstance(metadata_path, str) else metadata_path
    
    # Load all metadata files
    all_metadata = {}
    for path in metadata_paths:
        logging.info(f"Loading metadata file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_metadata[path] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading metadata file {path}: {e}")
            return
    
    first_metadata = list(all_metadata.values())[0]
    image_filenames = list(first_metadata.keys())
    image_paths = [os.path.join(images_folder, filename) for filename in image_filenames]
    
    logging.info(f"Processing {len(image_paths)} images")
    
    async with ImageAnalyzer(service) as analyzer:
        if analyzer.supports_batch:
            responses = await analyzer.analyze_batch(image_paths)
            results = dict(zip(image_filenames, responses))
        else:
            results = {}
            for filename, image_path in zip(image_filenames, image_paths):
                logging.info(f"Processing image: {filename}")
                prediction, analysis = await analyzer.analyze(image_path)
                results[filename] = (prediction, analysis)
    
    for path, metadata in all_metadata.items():
        logging.info(f"Updating metadata file: {path}")
        for filename, (prediction, analysis) in results.items():
            if filename in metadata:
                current_data = metadata[filename] if isinstance(metadata[filename], dict) else {"legacy_value": metadata[filename]}
                current_data["prediction_gpt"] = prediction
                current_data["analysis_gpt"] = analysis if analysis is not None else "Analysis not available"
                metadata[filename] = current_data
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved metadata to {path}")
        except Exception as e:
            logging.error(f"Failed to save metadata to {path}: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Process both metadata files
    metadata_paths = [
        "./btc_4h_dataset/metadata.json",
        "./btc_4h_dataset/metadata2.json"
    ]
    images_dir = "./btc_4h_dataset/plots"
    api_key = os.getenv("OPENAI_API_KEY")
    analysis_output = "./btc_4h_dataset/prediction_analysis.txt"

    asyncio.run(process_images_and_update_json(metadata_paths, images_dir, api_key, use_batch=True))
    logging.info("GPT analysis completed for both metadata files.")
    
    # Analyze predictions using the main metadata file
    analyzer = PredictionAnalyzer("./btc_4h_dataset/metadata.json")
    analyzer.print_analysis(analysis_output)
    
    logging.info("Script execution finished.")
