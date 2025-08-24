#!/usr/bin/env python3
"""
GPT Vision Analysis Core Module
==============================

This module provides GPT-powered analysis of trading charts using OpenAI's Vision API.
Refactored from the original gpt_analysis.py with improved structure and error handling.
"""

import asyncio
import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

from openai import OpenAI


@dataclass(frozen=True, slots=True)
class AnalysisPrompt:
    """Configuration for GPT analysis prompts."""
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
    """Abstract base class for GPT analysis services."""
    
    def __init__(self, prompt: AnalysisPrompt):
        self._prompt = prompt
    
    @abstractmethod
    async def analyze_image(self, image_path: str, image_b64: str) -> str:
        """Analyze an image and return JSON response."""
        pass
    
    @abstractmethod
    async def __aenter__(self):
        """Initialize resources when entering async context."""
        pass
        
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources when exiting async context."""
        pass


class GPTAnalysisService(AnalysisService):
    """GPT analysis service using OpenAI Vision API."""
    
    CALLS_PER_MINUTE = 50
    SLEEP_TIME = 60 / CALLS_PER_MINUTE 
    
    def __init__(self, prompt: AnalysisPrompt, api_key: str, model: str = "gpt-4o"):
        super().__init__(prompt)
        self._api_key = api_key
        self._client = None
        self._last_call_time: float = 0.0
        self._model = model
    
    async def __aenter__(self):
        """Initialize OpenAI client."""
        self._client = self._init_client(self._api_key)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """No special cleanup needed."""
        pass
    
    def _init_client(self, api_key: str) -> OpenAI:
        """Initialize OpenAI client."""
        if not api_key:
            raise EnvironmentError("OpenAI API key must be provided")
        
        try:
            client = OpenAI(api_key=api_key)
            logging.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def analyze_image(self, image_path: str, image_b64: str) -> str:
        """Analyze image using GPT Vision API."""
        if not self._client:
            raise RuntimeError("Service must be used within an async context manager")
            
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self.SLEEP_TIME:
            sleep_duration = self.SLEEP_TIME - time_since_last_call
            await asyncio.sleep(sleep_duration)
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
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
            
        except Exception as e:
            logging.error(f"GPT API call failed: {e}")
            raise


class GPTAnalyzer:
    """High-level GPT analyzer for trading charts."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize GPT analyzer.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.api_key = api_key
        self.model = model
        self.prompt = AnalysisPrompt()
        self._service = GPTAnalysisService(self.prompt, api_key, model)
        
        # Cache for recent predictions
        self._prediction_cache: Dict[str, Tuple[str, Optional[str], datetime]] = {}
        self._cache_duration_minutes = 5
        
        logging.info(f"GPTAnalyzer initialized with model: {model}")
    
    async def analyze_chart(self, chart_path: str) -> Tuple[str, Optional[str]]:
        """
        Analyze a trading chart and return prediction and analysis.
        
        Args:
            chart_path: Path to chart image
            
        Returns:
            Tuple of (prediction, analysis_text)
        """
        # Check cache first
        cached_result = self._get_cached_prediction(chart_path)
        if cached_result:
            logging.info(f"Using cached prediction for {chart_path}")
            return cached_result
        
        try:
            # Encode image
            image_b64 = self._encode_image_to_base64(chart_path)
            
            # Analyze with GPT
            async with self._service as service:
                response = await service.analyze_image(chart_path, image_b64)
            
            # Process response
            prediction, analysis_text = self._process_response(response, chart_path)
            
            # Cache result
            self._cache_prediction(chart_path, prediction, analysis_text)
            
            logging.info(f"GPT Analysis completed for {os.path.basename(chart_path)}")
            logging.info(f"  Prediction: {prediction}")
            
            return prediction, analysis_text
            
        except Exception as e:
            logging.error(f"Error analyzing chart {chart_path}: {e}")
            return "error_analysis_failed", None
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _process_response(self, raw_response: str, image_path: str) -> Tuple[str, Optional[str]]:
        """Process GPT response and extract prediction and analysis."""
        try:
            parsed_response = json.loads(raw_response)
            
            if not isinstance(parsed_response, dict):
                logging.error(f"Response is not a dictionary for {image_path}")
                return "error_json_not_dict", None
            
            analysis_text = parsed_response.get(self.prompt.analysis_key)
            prediction = parsed_response.get(self.prompt.prediction_key, "").lower()
            
            if not analysis_text:
                analysis_text = "No analysis provided by model"
                
            if prediction not in self.prompt.valid_predictions:
                logging.warning(f"Invalid prediction '{prediction}' for {image_path}")
                return "error_invalid_prediction", analysis_text
            
            return prediction, analysis_text
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response for {image_path}: {e}")
            return "error_json_decode", None
        except Exception as e:
            logging.error(f"Error processing response for {image_path}: {e}")
            return "error_processing_response", None
    
    def _get_cached_prediction(self, chart_path: str) -> Optional[Tuple[str, Optional[str]]]:
        """Get cached prediction if available and not expired."""
        if chart_path not in self._prediction_cache:
            return None
        
        prediction, analysis_text, timestamp = self._prediction_cache[chart_path]
        
        # Check if cache is expired
        now = datetime.now()
        if (now - timestamp).total_seconds() > (self._cache_duration_minutes * 60):
            del self._prediction_cache[chart_path]
            return None
        
        return prediction, analysis_text
    
    def _cache_prediction(self, chart_path: str, prediction: str, analysis_text: Optional[str]) -> None:
        """Cache a prediction result."""
        self._prediction_cache[chart_path] = (prediction, analysis_text, datetime.now())
        
        # Limit cache size
        if len(self._prediction_cache) > 100:
            oldest_key = min(self._prediction_cache.keys(), 
                           key=lambda k: self._prediction_cache[k][2])
            del self._prediction_cache[oldest_key]


class LiveGPTAnalyzer:
    """Live GPT analyzer for real-time trading with enhanced features."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize live GPT analyzer.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.analyzer = GPTAnalyzer(api_key, model)
        self._analysis_count = 0
        self._last_analysis_time: Optional[datetime] = None
        
        logging.info("LiveGPTAnalyzer initialized")
    
    async def analyze_and_generate_signals(self, chart_path: str) -> Dict[str, Union[str, bool]]:
        """
        Analyze chart and generate trading signals.
        
        Args:
            chart_path: Path to chart image
            
        Returns:
            Dictionary with prediction and trading signals
        """
        try:
            # Analyze chart
            prediction, analysis_text = await self.analyzer.analyze_chart(chart_path)
            
            # Generate trading signals
            signals = self._prediction_to_signals(prediction)
            
            # Update state
            self._analysis_count += 1
            self._last_analysis_time = datetime.now()
            
            # Create result
            result = {
                'prediction': prediction,
                'analysis': analysis_text or "Analysis not available",
                'timestamp': self._last_analysis_time.isoformat(),
                'analysis_count': self._analysis_count,
                **signals
            }
            
            logging.info(f"Live analysis #{self._analysis_count} completed: {prediction}")
            return result
            
        except Exception as e:
            logging.error(f"Error in live analysis: {e}")
            return {
                'prediction': 'error_analysis_failed',
                'analysis': f"Analysis failed: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'analysis_count': self._analysis_count,
                'enter_long': False,
                'exit_long': False,
                'enter_short': False,
                'exit_short': False
            }
    
    def _prediction_to_signals(self, prediction: str) -> Dict[str, bool]:
        """
        Convert GPT prediction to trading signals.
        
        Args:
            prediction: GPT prediction ('bullish', 'bearish', 'neutral', or error)
            
        Returns:
            Dictionary with trading signals
        """
        signals = {
            'enter_long': False,
            'exit_long': False,
            'enter_short': False,
            'exit_short': False
        }
        
        if prediction == 'bullish':
            signals['enter_long'] = True
            signals['exit_short'] = True  # Close short positions
        elif prediction == 'bearish':
            signals['enter_short'] = True
            signals['exit_long'] = True   # Close long positions
        elif prediction == 'neutral':
            # Close all positions on neutral prediction
            signals['exit_long'] = True
            signals['exit_short'] = True
        
        return signals
    
    def get_stats(self) -> Dict:
        """Get analyzer statistics."""
        return {
            'analysis_count': self._analysis_count,
            'last_analysis_time': self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            'cache_size': len(self.analyzer._prediction_cache),
            'model': self.analyzer.model
        }
