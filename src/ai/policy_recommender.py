"""
AI-powered policy recommendation engine for NeuroUrban.
Analyzes city data and suggests optimal policies for urban development.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.config.settings import Config

class PolicyRecommender:
    """AI-powered policy recommendation engine."""
    
    def __init__(self, config: Config):
        """
        Initialize the policy recommender.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Policy database
        self.policy_database = self._load_policy_database()
        
        # AI models
        self.sentiment_analyzer = None
        self.text_generator = None
        
        # Initialize AI models if available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_ai_models()
        
        self.logger.info("🤖 Policy recommender initialized")
    
    def recommend_policies(self, city_data: Dict, focus_areas: List[str] = None) -> Dict:
        """
        Recommend policies for a city based on its data and focus areas.
        
        Args:
            city_data: City statistics and features
            focus_areas: Optional list of focus areas (e.g., 'sustainability', 'transportation')
            
        Returns:
            Dictionary containing policy recommendations
        """
        self.logger.info(f"🔍 Generating policy recommendations for {city_data.get('name', 'Unknown City')}")
        
        # Analyze city strengths and weaknesses
        analysis = self._analyze_city_performance(city_data)
        
        # Generate targeted recommendations
        recommendations = self._generate_recommendations(analysis, focus_areas)
        
        # Predict policy impact
        impact_predictions = self._predict_policy_impact(city_data, recommendations)
        
        # Create comprehensive report
        policy_report = {
            'city_name': city_data.get('name', 'Unknown City'),
            'timestamp': datetime.now().isoformat(),
            'city_analysis': analysis,
            'recommendations': recommendations,
            'impact_predictions': impact_predictions,
            'implementation_timeline': self._create_implementation_timeline(recommendations),
            'success_metrics': self._define_success_metrics(recommendations)
        }
        
        self.logger.info(f"✅ Generated {len(recommendations)} policy recommendations")
        return policy_report
    
    def compare_policy_scenarios(self, city_data: Dict, scenarios: List[Dict]) -> Dict:
        """
        Compare different policy scenarios and their predicted outcomes.
        
        Args:
            city_data: City statistics and features
            scenarios: List of policy scenario dictionaries
            
        Returns:
            Comparison results with rankings and predictions
        """
        scenario_results = []
        
        for i, scenario in enumerate(scenarios):
            # Predict outcomes for this scenario
            outcomes = self._predict_scenario_outcomes(city_data, scenario)
            
            scenario_results.append({
                'scenario_id': i + 1,
                'name': scenario.get('name', f'Scenario {i + 1}'),
                'policies': scenario.get('policies', []),
                'predicted_outcomes': outcomes,
                'overall_score': outcomes.get('overall_score', 0),
                'risk_level': self._assess_risk_level(outcomes),
                'implementation_cost': self._estimate_implementation_cost(scenario)
            })
        
        # Rank scenarios by overall score
        scenario_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'comparison_timestamp': datetime.now().isoformat(),
            'scenarios': scenario_results,
            'recommended_scenario': scenario_results[0] if scenario_results else None,
            'key_insights': self._generate_scenario_insights(scenario_results)
        }
    
    def _analyze_city_performance(self, city_data: Dict) -> Dict:
        """Analyze city performance across different dimensions."""
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'threats': [],
            'performance_scores': {},
            'priority_areas': []
        }
        
        # Define performance categories and their indicators
        categories = {
            'sustainability': ['co2_emissions_per_capita', 'green_space_percentage', 'renewable_energy_usage'],
            'transportation': ['walkability_score', 'public_transport_coverage', 'cycling_infrastructure_km'],
            'economy': ['gdp_per_capita', 'unemployment_rate', 'innovation_index'],
            'livability': ['livability_index', 'safety_index', 'air_quality_index'],
            'infrastructure': ['internet_speed_mbps', 'hospitals_count', 'schools_count'],
            'governance': ['ease_of_doing_business', 'corruption_index', 'digital_government_index']
        }
        
        # Calculate performance scores
        for category, indicators in categories.items():
            scores = []
            for indicator in indicators:
                value = self._extract_nested_value(city_data, indicator)
                if value is not None:
                    # Normalize to 0-100 scale (simplified)
                    normalized = min(100, max(0, value))
                    scores.append(normalized)
            
            if scores:
                avg_score = np.mean(scores)
                analysis['performance_scores'][category] = avg_score
                
                # Classify as strength or weakness
                if avg_score >= 70:
                    analysis['strengths'].append({
                        'category': category,
                        'score': avg_score,
                        'description': f"Strong performance in {category}"
                    })
                elif avg_score <= 40:
                    analysis['weaknesses'].append({
                        'category': category,
                        'score': avg_score,
                        'description': f"Needs improvement in {category}"
                    })
        
        # Identify priority areas (lowest scoring categories)
        sorted_categories = sorted(analysis['performance_scores'].items(), key=lambda x: x[1])
        analysis['priority_areas'] = [cat for cat, score in sorted_categories[:3]]
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict, focus_areas: List[str] = None) -> List[Dict]:
        """Generate policy recommendations based on analysis."""
        recommendations = []
        
        # Focus on weaknesses and priority areas
        target_areas = focus_areas if focus_areas else analysis['priority_areas']
        
        for area in target_areas:
            if area in self.policy_database:
                area_policies = self.policy_database[area]
                
                # Select most relevant policies
                for policy in area_policies[:3]:  # Top 3 policies per area
                    recommendation = {
                        'policy_id': policy['id'],
                        'title': policy['title'],
                        'category': area,
                        'description': policy['description'],
                        'implementation_steps': policy['implementation_steps'],
                        'expected_outcomes': policy['expected_outcomes'],
                        'timeline': policy['timeline'],
                        'cost_estimate': policy['cost_estimate'],
                        'success_examples': policy.get('success_examples', []),
                        'priority_level': self._calculate_priority_level(area, analysis),
                        'feasibility_score': policy.get('feasibility_score', 70)
                    }
                    recommendations.append(recommendation)
        
        # Sort by priority level
        recommendations.sort(key=lambda x: x['priority_level'], reverse=True)
        
        return recommendations
    
    def _predict_policy_impact(self, city_data: Dict, recommendations: List[Dict]) -> Dict:
        """Predict the impact of recommended policies."""
        impact_predictions = {
            'short_term': {},  # 1-2 years
            'medium_term': {},  # 3-5 years
            'long_term': {},   # 5+ years
            'overall_impact_score': 0,
            'confidence_level': 'medium'
        }
        
        # Simulate impact for each recommendation
        total_impact = 0
        for rec in recommendations:
            category = rec['category']
            
            # Estimate impact based on policy type and city characteristics
            base_impact = rec.get('expected_impact', 15)  # Default 15% improvement
            
            # Adjust based on city's current performance
            current_score = city_data.get(f'{category}_score', 50)
            if current_score < 30:
                impact_multiplier = 1.5  # Higher impact for low-performing areas
            elif current_score > 70:
                impact_multiplier = 0.7  # Lower impact for already good areas
            else:
                impact_multiplier = 1.0
            
            adjusted_impact = base_impact * impact_multiplier
            total_impact += adjusted_impact
            
            # Distribute impact across time periods
            impact_predictions['short_term'][category] = adjusted_impact * 0.3
            impact_predictions['medium_term'][category] = adjusted_impact * 0.5
            impact_predictions['long_term'][category] = adjusted_impact * 0.2
        
        impact_predictions['overall_impact_score'] = min(100, total_impact)
        
        return impact_predictions
    
    def _load_policy_database(self) -> Dict:
        """Load the policy database with predefined policies."""
        return {
            'sustainability': [
                {
                    'id': 'sus_001',
                    'title': 'Green Building Standards',
                    'description': 'Implement mandatory green building certification for new constructions',
                    'implementation_steps': [
                        'Develop green building standards',
                        'Create certification process',
                        'Train inspectors',
                        'Launch incentive programs'
                    ],
                    'expected_outcomes': ['20% reduction in building energy consumption', 'Improved air quality'],
                    'timeline': '18 months',
                    'cost_estimate': 'Medium',
                    'feasibility_score': 75
                },
                {
                    'id': 'sus_002',
                    'title': 'Urban Forest Initiative',
                    'description': 'Increase urban tree coverage by 30% over 5 years',
                    'implementation_steps': [
                        'Identify planting locations',
                        'Source native tree species',
                        'Community planting programs',
                        'Maintenance planning'
                    ],
                    'expected_outcomes': ['Improved air quality', 'Reduced urban heat island effect'],
                    'timeline': '5 years',
                    'cost_estimate': 'High',
                    'feasibility_score': 85
                }
            ],
            'transportation': [
                {
                    'id': 'trans_001',
                    'title': 'Bike Lane Network Expansion',
                    'description': 'Create comprehensive protected bike lane network',
                    'implementation_steps': [
                        'Conduct traffic analysis',
                        'Design protected lanes',
                        'Install infrastructure',
                        'Launch bike-share program'
                    ],
                    'expected_outcomes': ['25% increase in cycling', 'Reduced traffic congestion'],
                    'timeline': '2 years',
                    'cost_estimate': 'Medium',
                    'feasibility_score': 80
                }
            ],
            'economy': [
                {
                    'id': 'econ_001',
                    'title': 'Innovation District Development',
                    'description': 'Establish dedicated innovation and startup districts',
                    'implementation_steps': [
                        'Identify suitable locations',
                        'Create zoning incentives',
                        'Develop infrastructure',
                        'Attract anchor institutions'
                    ],
                    'expected_outcomes': ['Job creation', 'Increased innovation index'],
                    'timeline': '3 years',
                    'cost_estimate': 'High',
                    'feasibility_score': 65
                }
            ]
        }
    
    def _initialize_ai_models(self):
        """Initialize AI models for advanced analysis."""
        try:
            # Initialize sentiment analysis for policy feedback
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                              model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # Initialize text generation for policy descriptions
            self.text_generator = pipeline("text-generation", 
                                         model="gpt2",
                                         max_length=100,
                                         num_return_sequences=1)
            
            self.logger.info("✅ AI models initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize AI models: {e}")
    
    def _extract_nested_value(self, data: Dict, key: str) -> Optional[float]:
        """Extract value from nested dictionary structure."""
        # Try direct access first
        if key in data:
            return data[key]
        
        # Try nested access
        for category in ['economy', 'environment', 'infrastructure', 'transportation', 'safety']:
            if category in data and isinstance(data[category], dict):
                if key in data[category]:
                    return data[category][key]
        
        return None
    
    def _calculate_priority_level(self, area: str, analysis: Dict) -> int:
        """Calculate priority level for a policy area."""
        score = analysis['performance_scores'].get(area, 50)
        
        # Lower scores get higher priority
        if score <= 30:
            return 90
        elif score <= 50:
            return 70
        elif score <= 70:
            return 50
        else:
            return 30
    
    def _predict_scenario_outcomes(self, city_data: Dict, scenario: Dict) -> Dict:
        """Predict outcomes for a policy scenario."""
        # Simplified prediction model
        base_score = 50
        policy_impact = len(scenario.get('policies', [])) * 5
        
        return {
            'overall_score': min(100, base_score + policy_impact),
            'sustainability_improvement': np.random.uniform(10, 30),
            'economic_growth': np.random.uniform(5, 20),
            'livability_increase': np.random.uniform(8, 25),
            'implementation_complexity': np.random.uniform(30, 80)
        }
    
    def _assess_risk_level(self, outcomes: Dict) -> str:
        """Assess risk level of policy implementation."""
        complexity = outcomes.get('implementation_complexity', 50)
        
        if complexity > 70:
            return 'High'
        elif complexity > 40:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_implementation_cost(self, scenario: Dict) -> str:
        """Estimate implementation cost for a scenario."""
        num_policies = len(scenario.get('policies', []))
        
        if num_policies > 5:
            return 'High'
        elif num_policies > 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_implementation_timeline(self, recommendations: List[Dict]) -> Dict:
        """Create implementation timeline for recommendations."""
        timeline = {
            'phase_1': [],  # 0-6 months
            'phase_2': [],  # 6-18 months
            'phase_3': []   # 18+ months
        }
        
        for rec in recommendations:
            if rec['priority_level'] > 80:
                timeline['phase_1'].append(rec['title'])
            elif rec['priority_level'] > 60:
                timeline['phase_2'].append(rec['title'])
            else:
                timeline['phase_3'].append(rec['title'])
        
        return timeline
    
    def _define_success_metrics(self, recommendations: List[Dict]) -> List[Dict]:
        """Define success metrics for policy recommendations."""
        metrics = []
        
        for rec in recommendations:
            metric = {
                'policy': rec['title'],
                'category': rec['category'],
                'kpis': rec.get('expected_outcomes', []),
                'measurement_frequency': 'Quarterly',
                'target_improvement': '15-25%'
            }
            metrics.append(metric)
        
        return metrics
    
    def _generate_scenario_insights(self, scenario_results: List[Dict]) -> List[str]:
        """Generate insights from scenario comparison."""
        insights = []
        
        if scenario_results:
            best_scenario = scenario_results[0]
            insights.append(f"Recommended scenario: {best_scenario['name']} with {best_scenario['overall_score']:.1f}% overall score")
            
            # Risk analysis
            high_risk_scenarios = [s for s in scenario_results if s['risk_level'] == 'High']
            if high_risk_scenarios:
                insights.append(f"{len(high_risk_scenarios)} scenarios have high implementation risk")
            
            # Cost analysis
            low_cost_scenarios = [s for s in scenario_results if s['implementation_cost'] == 'Low']
            if low_cost_scenarios:
                insights.append(f"{len(low_cost_scenarios)} scenarios are low-cost to implement")
        
        return insights
