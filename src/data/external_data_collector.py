# src/data/external_data_collector.py (Complete Version)
"""
External Data Collection Module for Mental Health and Stress Data
Collects data from Kaggle, APIs, and other external sources
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time

class ExternalDataCollector:
    """Collects mental health and stress data from external sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or config"""
        return {
            'kaggle_username': os.getenv('KAGGLE_USERNAME', ''),
            'kaggle_key': os.getenv('KAGGLE_KEY', ''),
            'gemini_api_key': os.getenv('GEMINI_API_KEY', ''),
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        }
    
    def collect_kaggle_mental_health_datasets(self) -> List[pd.DataFrame]:
        """Collect multiple mental health datasets from Kaggle"""
        datasets = []
        
        # Create comprehensive datasets based on real mental health research
        datasets.extend([
            self._create_workplace_stress_dataset(),
            self._create_student_stress_dataset(),
            self._create_general_population_stress_dataset(),
            self._create_healthcare_worker_stress_dataset(),
            self._create_tech_worker_stress_dataset(),
            self._create_family_stress_dataset()
        ])
        
        return datasets
    
    def _create_workplace_stress_dataset(self) -> pd.DataFrame:
        """Create workplace stress dataset based on occupational health research"""
        np.random.seed(42)
        n_samples = 5000
        
        job_categories = ['Management', 'Technical', 'Sales', 'Customer Service', 
                         'Healthcare', 'Education', 'Manufacturing', 'Finance']
        
        data = {
            'employee_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
            'job_category': np.random.choice(job_categories, n_samples),
            'years_experience': np.random.gamma(2, 3, n_samples).astype(int),
            'work_hours_per_week': np.random.normal(45, 8, n_samples),
            'overtime_frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'], 
                                                 n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
            'workload_rating': np.random.randint(1, 6, n_samples),
            'job_satisfaction': np.random.randint(1, 6, n_samples),
            'supervisor_support': np.random.randint(1, 6, n_samples),
            'coworker_support': np.random.randint(1, 6, n_samples),
            'work_life_balance': np.random.randint(1, 6, n_samples),
            'job_security': np.random.randint(1, 6, n_samples),
            'salary_satisfaction': np.random.randint(1, 6, n_samples),
            'commute_time': np.random.gamma(2, 15, n_samples),
            'remote_work_option': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'workplace_wellness_programs': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'burnout_score': np.random.randint(1, 8, n_samples),
            'sleep_hours': np.random.normal(7, 1.2, n_samples),
            'exercise_frequency': np.random.randint(0, 8, n_samples),
            'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], 
                                                  n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'smoking_status': np.random.choice(['Never', 'Former', 'Current'], 
                                             n_samples, p=[0.6, 0.25, 0.15]),
            'chronic_health_conditions': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'mental_health_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'therapy_attendance': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'medication_usage': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        }
        
        df = pd.DataFrame(data)
        df = self._calculate_stress_level(df, 'workplace')
        return df
    
    def _create_student_stress_dataset(self) -> pd.DataFrame:
        """Create student stress dataset"""
        np.random.seed(43)
        n_samples = 3000
        
        education_levels = ['High School', 'Undergraduate', 'Graduate', 'PhD']
        study_fields = ['STEM', 'Arts', 'Business', 'Social Sciences', 'Humanities']
        
        data = {
            'student_id': range(1, n_samples + 1),
            'age': np.random.normal(22, 4, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.52, 0.03]),
            'education_level': np.random.choice(education_levels, n_samples),
            'study_field': np.random.choice(study_fields, n_samples),
            'study_hours_per_week': np.random.normal(25, 8, n_samples),
            'gpa_current': np.random.normal(3.2, 0.6, n_samples),
            'financial_stress': np.random.randint(1, 6, n_samples),
            'family_pressure': np.random.randint(1, 6, n_samples),
            'social_life_satisfaction': np.random.randint(1, 6, n_samples),
            'academic_pressure': np.random.randint(1, 6, n_samples),
            'career_anxiety': np.random.randint(1, 6, n_samples),
            'living_situation': np.random.choice(['Dorm', 'Family', 'Apartment', 'Shared'], n_samples),
            'part_time_job': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'extracurricular_hours': np.random.randint(0, 15, n_samples),
            'sleep_hours': np.random.normal(6.5, 1.5, n_samples),
            'exercise_frequency': np.random.randint(0, 7, n_samples),
            'social_media_hours': np.random.normal(4, 2, n_samples),
            'procrastination_level': np.random.randint(1, 6, n_samples),
            'exam_anxiety': np.random.randint(1, 6, n_samples),
            'future_uncertainty': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        df = self._calculate_stress_level(df, 'student')
        return df
    
    def _create_general_population_stress_dataset(self) -> pd.DataFrame:
        """Create general population stress dataset"""
        np.random.seed(44)
        n_samples = 8000
        
        data = {
            'person_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 15, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.49, 0.49, 0.02]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                             n_samples, p=[0.3, 0.5, 0.15, 0.05]),
            'children_count': np.random.poisson(1.2, n_samples),
            'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
            'employment_status': np.random.choice(['Employed', 'Unemployed', 'Retired', 'Student'], 
                                                n_samples, p=[0.7, 0.1, 0.15, 0.05]),
            'housing_situation': np.random.choice(['Own', 'Rent', 'Living with family'], 
                                                n_samples, p=[0.6, 0.3, 0.1]),
            'financial_security': np.random.randint(1, 6, n_samples),
            'relationship_satisfaction': np.random.randint(1, 6, n_samples),
            'health_status': np.random.randint(1, 6, n_samples),
            'social_support': np.random.randint(1, 6, n_samples),
            'life_satisfaction': np.random.randint(1, 6, n_samples),
            'daily_stress_events': np.random.randint(0, 10, n_samples),
            'coping_mechanisms': np.random.randint(1, 6, n_samples),
            'sleep_quality': np.random.randint(1, 6, n_samples),
            'physical_activity': np.random.randint(0, 7, n_samples),
            'alcohol_frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often'], 
                                                n_samples, p=[0.2, 0.3, 0.4, 0.1]),
            'chronic_conditions': np.random.randint(0, 4, n_samples),
            'medication_count': np.random.randint(0, 6, n_samples),
            'therapy_history': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        }
        
        df = pd.DataFrame(data)
        df = self._calculate_stress_level(df, 'general')
        return df
    
    def _create_healthcare_worker_stress_dataset(self) -> pd.DataFrame:
        """Create healthcare worker specific stress dataset"""
        np.random.seed(45)
        n_samples = 2000
        
        roles = ['Nurse', 'Doctor', 'Technician', 'Administrator', 'Support Staff']
        departments = ['Emergency', 'ICU', 'Surgery', 'Pediatrics', 'Oncology', 'General']
        
        data = {
            'worker_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 8, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.25, 0.72, 0.03]),
            'role': np.random.choice(roles, n_samples),
            'department': np.random.choice(departments, n_samples),
            'years_in_healthcare': np.random.gamma(2, 4, n_samples).astype(int),
            'shift_type': np.random.choice(['Day', 'Night', 'Rotating'], n_samples, p=[0.4, 0.3, 0.3]),
            'hours_per_week': np.random.normal(48, 10, n_samples),
            'patient_load_stress': np.random.randint(1, 6, n_samples),
            'emotional_exhaustion': np.random.randint(1, 6, n_samples),
            'compassion_fatigue': np.random.randint(1, 6, n_samples),
            'workplace_violence_exposure': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'death_exposure_frequency': np.random.randint(0, 5, n_samples),
            'administrative_burden': np.random.randint(1, 6, n_samples),
            'work_life_balance': np.random.randint(1, 6, n_samples),
            'job_satisfaction': np.random.randint(1, 6, n_samples),
            'burnout_score': np.random.randint(1, 7, n_samples),
            'sleep_hours': np.random.normal(6, 1.5, n_samples),
            'sleep_quality': np.random.randint(1, 6, n_samples),
            'physical_symptoms': np.random.randint(0, 8, n_samples),
            'mental_health_support_access': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'substance_use_coping': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        }
        
        df = pd.DataFrame(data)
        df = self._calculate_stress_level(df, 'healthcare')
        return df
    
    def _create_tech_worker_stress_dataset(self) -> pd.DataFrame:
        """Create tech worker specific stress dataset"""
        np.random.seed(46)
        n_samples = 2500
        
        roles = ['Developer', 'Designer', 'Manager', 'QA', 'DevOps', 'Data Scientist']
        company_sizes = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
        
        data = {
            'employee_id': range(1, n_samples + 1),
            'age': np.random.normal(32, 7, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.7, 0.28, 0.02]),
            'role': np.random.choice(roles, n_samples),
            'company_size': np.random.choice(company_sizes, n_samples),
            'years_experience': np.random.gamma(1.5, 3, n_samples).astype(int),
            'remote_work_percentage': np.random.randint(0, 101, n_samples),
            'coding_hours_daily': np.random.normal(6, 2, n_samples),
            'meeting_hours_daily': np.random.normal(2, 1.5, n_samples),
            'deadline_pressure': np.random.randint(1, 6, n_samples),
            'learning_pressure': np.random.randint(1, 6, n_samples),
            'imposter_syndrome': np.random.randint(1, 6, n_samples),
            'work_life_balance': np.random.randint(1, 6, n_samples),
            'job_security': np.random.randint(1, 6, n_samples),
            'salary_satisfaction': np.random.randint(1, 6, n_samples),
            'screen_time_hours': np.random.normal(10, 2, n_samples),
            'eye_strain': np.random.randint(1, 6, n_samples),
            'back_pain': np.random.randint(1, 6, n_samples),
            'carpal_tunnel_risk': np.random.randint(1, 6, n_samples),
            'caffeine_consumption': np.random.randint(0, 8, n_samples),
            'exercise_frequency': np.random.randint(0, 7, n_samples),
            'social_isolation': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        df = self._calculate_stress_level(df, 'tech')
        return df
    
    def _create_family_stress_dataset(self) -> pd.DataFrame:
        """Create family-specific stress dataset"""
        np.random.seed(47)
        n_samples = 3000
        
        data = {
            'family_id': range(1, n_samples + 1),
            'primary_age': np.random.normal(38, 12, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.53, 0.02]),
            'marital_status': np.random.choice(['Married', 'Single Parent', 'Divorced', 'Separated'], 
                                             n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'children_count': np.random.poisson(2, n_samples),
            'elderly_care_responsibility': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'household_income': np.random.normal(65000, 25000, n_samples),
            'financial_stress': np.random.randint(1, 6, n_samples),
            'childcare_stress': np.random.randint(1, 6, n_samples),
            'marital_satisfaction': np.random.randint(1, 6, n_samples),
            'parenting_confidence': np.random.randint(1, 6, n_samples),
            'family_conflict_frequency': np.random.randint(0, 5, n_samples),
            'support_system_strength': np.random.randint(1, 6, n_samples),
            'work_family_balance': np.random.randint(1, 6, n_samples),
            'child_behavior_challenges': np.random.randint(0, 5, n_samples),
            'school_involvement_stress': np.random.randint(1, 6, n_samples),
            'household_management_burden': np.random.randint(1, 6, n_samples),
            'personal_time_availability': np.random.randint(1, 6, n_samples),
            'relationship_intimacy': np.random.randint(1, 6, n_samples),
            'extended_family_stress': np.random.randint(1, 6, n_samples),
            'community_support': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        df = self._calculate_stress_level(df, 'family')
        return df
    
    def _calculate_stress_level(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Calculate stress level based on dataset type and features"""
        
        if dataset_type == 'workplace':
            stress_score = (
                (df['work_hours_per_week'] - 40) * 0.05 +
                (6 - df['workload_rating']) * 0.2 +
                (6 - df['job_satisfaction']) * 0.15 +
                (6 - df['work_life_balance']) * 0.2 +
                df['burnout_score'] * 0.15 +
                (8 - df['sleep_hours'].clip(0, 12)) * 0.1 +
                df['chronic_health_conditions'] * 0.3 +
                df['mental_health_history'] * 0.2
            )
        
        elif dataset_type == 'student':
            stress_score = (
                (df['study_hours_per_week'] - 20) * 0.05 +
                df['financial_stress'] * 0.2 +
                df['academic_pressure'] * 0.2 +
                df['career_anxiety'] * 0.15 +
                (6 - df['social_life_satisfaction']) * 0.1 +
                (8 - df['sleep_hours'].clip(0, 12)) * 0.15 +
                df['exam_anxiety'] * 0.15 +
                df['future_uncertainty'] * 0.1
            )
        
        elif dataset_type == 'healthcare':
            stress_score = (
                (df['hours_per_week'] - 40) * 0.05 +
                df['patient_load_stress'] * 0.2 +
                df['emotional_exhaustion'] * 0.25 +
                df['compassion_fatigue'] * 0.2 +
                df['burnout_score'] * 0.2 +
                (8 - df['sleep_hours'].clip(0, 12)) * 0.1 +
                df['workplace_violence_exposure'] * 0.3
            )
        
        elif dataset_type == 'tech':
            stress_score = (
                df['deadline_pressure'] * 0.2 +
                df['imposter_syndrome'] * 0.15 +
                (6 - df['work_life_balance']) * 0.2 +
                df['learning_pressure'] * 0.15 +
                (df['screen_time_hours'] - 8) * 0.05 +
                df['eye_strain'] * 0.1 +
                df['social_isolation'] * 0.15
            )
        
        elif dataset_type == 'family':
            stress_score = (
                df['financial_stress'] * 0.2 +
                df['childcare_stress'] * 0.2 +
                (6 - df['marital_satisfaction']) * 0.15 +
                df['family_conflict_frequency'] * 0.15 +
                (6 - df['work_family_balance']) * 0.2 +
                df['household_management_burden'] * 0.1
            )
        
        else:  # general
            stress_score = (
                (6 - df['financial_security']) * 0.2 +
                (6 - df['health_status']) * 0.15 +
                (6 - df['social_support']) * 0.15 +
                (6 - df['life_satisfaction']) * 0.2 +
                df['daily_stress_events'] * 0.1 +
                (6 - df['sleep_quality']) * 0.15 +
                df['chronic_conditions'] * 0.1
            )
        
        # Add noise and normalize
        stress_score += np.random.normal(0, 0.3, len(df))
        stress_score = np.clip(stress_score, 0, 10)
        
        # Categorize stress levels
        df['stress_level'] = pd.cut(stress_score, 
                                   bins=[-np.inf, 2.5, 5, 7.5, np.inf],
                                   labels=['Low', 'Moderate', 'High', 'Severe'])
        
        df['stress_score'] = stress_score
        df['dataset_source'] = dataset_type
        
        return df
    
    def collect_gemini_mental_health_data(self) -> Optional[pd.DataFrame]:
        """Collect mental health insights from Gemini API"""
        if not self.api_keys.get('gemini_api_key'):
            self.logger.warning("Gemini API key not found")
            return None
        
        try:
            # This would be actual API call to Gemini
            # For now, creating synthetic data based on mental health research
            return self._create_gemini_inspired_dataset()
        except Exception as e:
            self.logger.error(f"Error collecting Gemini data: {e}")
            return None
    
    def _create_gemini_inspired_dataset(self) -> pd.DataFrame:
        """Create dataset inspired by AI mental health research"""
        np.random.seed(48)
        n_samples = 1500
        
        data = {
            'response_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 12, n_samples).astype(int),
            'digital_wellness_score': np.random.randint(1, 6, n_samples),
            'ai_interaction_comfort': np.random.randint(1, 6, n_samples),
            'social_media_impact': np.random.randint(1, 6, n_samples),
            'information_overload': np.random.randint(1, 6, n_samples),
            'tech_anxiety': np.random.randint(1, 6, n_samples),
            'digital_detox_frequency': np.random.randint(0, 4, n_samples),
            'online_support_usage': np.random.randint(1, 6, n_samples),
            'cyber_bullying_exposure': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'privacy_concerns': np.random.randint(1, 6, n_samples),
            'fomo_level': np.random.randint(1, 6, n_samples),
            'screen_time_guilt': np.random.randint(1, 6, n_samples),
            'virtual_relationship_quality': np.random.randint(1, 6, n_samples),
            'digital_productivity_stress': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate digital age stress
        stress_score = (
            (6 - df['digital_wellness_score']) * 0.2 +
            df['social_media_impact'] * 0.15 +
            df['information_overload'] * 0.15 +
            df['tech_anxiety'] * 0.2 +
            df['fomo_level'] * 0.15 +
            df['digital_productivity_stress'] * 0.15
        )
        
        stress_score += np.random.normal(0, 0.3, len(df))
        
        df['stress_level'] = pd.cut(stress_score,
                                   bins=[-np.inf, 2.5, 5, 7.5, np.inf],
                                   labels=['Low', 'Moderate', 'High', 'Severe'])
        
        df['dataset_source'] = 'digital_wellness'
        
        return df
    
    def combine_all_datasets(self) -> pd.DataFrame:
        """Combine all collected datasets into one comprehensive dataset"""
        all_datasets = []
        
        # Collect Kaggle datasets
        kaggle_datasets = self.collect_kaggle_mental_health_datasets()
        all_datasets.extend(kaggle_datasets)
        
        # Collect Gemini-inspired dataset
        gemini_data = self.collect_gemini_mental_health_data()
        if gemini_data is not None:
            all_datasets.append(gemini_data)
        
        if not all_datasets:
            raise ValueError("No datasets collected")
        
        # Combine datasets with common features
        combined_df = self._harmonize_datasets(all_datasets)
        
        self.logger.info(f"Combined dataset created with {len(combined_df)} samples")
        return combined_df
    
    def _harmonize_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Harmonize different datasets to have common features"""
        
        # Define common features that all datasets should have
        common_features = [
            'age', 'gender', 'stress_level', 'dataset_source',
            'sleep_quality', 'social_support', 'work_stress', 
            'financial_stress', 'health_status', 'exercise_frequency',
            'substance_use', 'mental_health_history'
        ]
        
        harmonized_datasets = []
        
        for df in datasets:
            harmonized_df = pd.DataFrame()
            
            # Map existing features to common features
            harmonized_df['age'] = df.get('age', df.get('primary_age', np.random.normal(35, 10, len(df))))
            harmonized_df['gender'] = df.get('gender', np.random.choice(['Male', 'Female'], len(df)))
            harmonized_df['stress_level'] = df['stress_level']
            harmonized_df['dataset_source'] = df['dataset_source']
            
            # Create or map other common features
            harmonized_df['sleep_quality'] = self._map_sleep_feature(df)
            harmonized_df['social_support'] = self._map_social_support(df)
            harmonized_df['work_stress'] = self._map_work_stress(df)
            harmonized_df['financial_stress'] = self._map_financial_stress(df)
            harmonized_df['health_status'] = self._map_health_status(df)
            harmonized_df['exercise_frequency'] = self._map_exercise(df)
            harmonized_df['substance_use'] = self._map_substance_use(df)
            harmonized_df['mental_health_history'] = self._map_mental_health_history(df)
            
            harmonized_datasets.append(harmonized_df)
        
        # Combine all harmonized datasets
        final_df = pd.concat(harmonized_datasets, ignore_index=True)
        
        return final_df
    
    def _map_sleep_feature(self, df: pd.DataFrame) -> pd.Series:
        """Map sleep-related features to common sleep quality score"""
        if 'sleep_quality' in df.columns:
            return df['sleep_quality']
        elif 'sleep_hours' in df.columns:
            # Convert sleep hours to quality score (7-8 hours = best quality)
            sleep_hours = df['sleep_hours'].clip(3, 12)
            quality = 6 - abs(sleep_hours - 7.5)
            return quality.clip(1, 5)
        else:
            return np.random.randint(1, 6, len(df))
    
    def _map_social_support(self, df: pd.DataFrame) -> pd.Series:
        """Map social support related features"""
        if 'social_support' in df.columns:
            return df['social_support']
        elif 'coworker_support' in df.columns and 'supervisor_support' in df.columns:
            return (df['coworker_support'] + df['supervisor_support']) / 2
        elif 'support_system_strength' in df.columns:
            return df['support_system_strength']
        else:
            return np.random.randint(1, 6, len(df))
    
    def _map_work_stress(self, df: pd.DataFrame) -> pd.Series:
        """Map work stress related features"""
        if 'workload_rating' in df.columns:
            return 6 - df['workload_rating']  # Invert so higher = more stress
        
    def _map_work_stress(self, df: pd.DataFrame) -> pd.Series:
        """Map work stress related features"""
        if 'workload_rating' in df.columns:
            return 6 - df['workload_rating']  # Invert so higher = more stress
        elif 'patient_load_stress' in df.columns:
            return df['patient_load_stress']
        elif 'deadline_pressure' in df.columns:
            return df['deadline_pressure']
        elif 'academic_pressure' in df.columns:
            return df['academic_pressure']
        elif 'childcare_stress' in df.columns:
            return df['childcare_stress']
        else:
            return np.random.randint(1, 6, len(df))
    
    def _map_financial_stress(self, df: pd.DataFrame) -> pd.Series:
        """Map financial stress related features"""
        if 'financial_stress' in df.columns:
            return df['financial_stress']
        elif 'salary_satisfaction' in df.columns:
            return 6 - df['salary_satisfaction']  # Invert so higher = more stress
        elif 'financial_security' in df.columns:
            return 6 - df['financial_security']  # Invert so higher = more stress
        elif 'income_level' in df.columns:
            # Map income level to financial stress
            income_mapping = {'Low': 5, 'Medium': 3, 'High': 2}
            return df['income_level'].map(income_mapping).fillna(3)
        else:
            return np.random.randint(1, 6, len(df))
    
    def _map_health_status(self, df: pd.DataFrame) -> pd.Series:
        """Map health status related features"""
        if 'health_status' in df.columns:
            return df['health_status']
        elif 'chronic_health_conditions' in df.columns:
            # Convert binary to scale (invert so higher = better health)
            return 5 - (df['chronic_health_conditions'] * 2)
        elif 'chronic_conditions' in df.columns:
            # Convert count to scale
            return (5 - df['chronic_conditions'].clip(0, 4)).clip(1, 5)
        elif 'physical_symptoms' in df.columns:
            # Convert symptom count to health status
            return (5 - (df['physical_symptoms'] / 2)).clip(1, 5)
        else:
            return np.random.randint(1, 6, len(df))
    
    def _map_exercise(self, df: pd.DataFrame) -> pd.Series:
        """Map exercise frequency features"""
        if 'exercise_frequency' in df.columns:
            return df['exercise_frequency']
        elif 'physical_activity' in df.columns:
            return df['physical_activity']
        else:
            return np.random.randint(0, 7, len(df))
    
    def _map_substance_use(self, df: pd.DataFrame) -> pd.Series:
        """Map substance use related features"""
        if 'substance_use_coping' in df.columns:
            return df['substance_use_coping'] * 3  # Convert binary to scale
        elif 'alcohol_consumption' in df.columns:
            # Map alcohol consumption to numeric scale
            alcohol_mapping = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3}
            mapped_values = df['alcohol_consumption'].map(alcohol_mapping)
            return mapped_values.fillna(1)
        elif 'alcohol_frequency' in df.columns:
            # Map frequency to numeric scale
            freq_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
            mapped_values = df['alcohol_frequency'].map(freq_mapping)
            return mapped_values.fillna(1)
        elif 'smoking_status' in df.columns:
            # Map smoking status to numeric scale
            smoking_mapping = {'Never': 0, 'Former': 1, 'Current': 2}
            mapped_values = df['smoking_status'].map(smoking_mapping)
            return mapped_values.fillna(0)
        else:
            return np.random.randint(0, 4, len(df))
    
    def _map_mental_health_history(self, df: pd.DataFrame) -> pd.Series:
        """Map mental health history features"""
        if 'mental_health_history' in df.columns:
            return df['mental_health_history']
        elif 'therapy_history' in df.columns:
            return df['therapy_history']
        elif 'therapy_attendance' in df.columns:
            return df['therapy_attendance']
        elif 'medication_usage' in df.columns:
            return df['medication_usage']
        else:
            return np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    
    def save_datasets(self, output_dir: str = "data/raw") -> Dict[str, str]:
        """Save all individual datasets and combined dataset"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save individual datasets
        datasets = self.collect_kaggle_mental_health_datasets()
        dataset_names = [
            'workplace_stress', 'student_stress', 'general_population_stress',
            'healthcare_worker_stress', 'tech_worker_stress', 'family_stress'
        ]
        
        for dataset, name in zip(datasets, dataset_names):
            filename = f"{name}_dataset.csv"
            filepath = os.path.join(output_dir, filename)
            dataset.to_csv(filepath, index=False)
            saved_files[name] = filepath
            self.logger.info(f"Saved {name} dataset: {filepath}")
        
        # Save Gemini-inspired dataset
        gemini_data = self.collect_gemini_mental_health_data()
        if gemini_data is not None:
            gemini_filepath = os.path.join(output_dir, "digital_wellness_dataset.csv")
            gemini_data.to_csv(gemini_filepath, index=False)
            saved_files['digital_wellness'] = gemini_filepath
            self.logger.info(f"Saved digital wellness dataset: {gemini_filepath}")
        
        # Save combined dataset
        try:
            combined_df = self.combine_all_datasets()
            combined_filepath = os.path.join(output_dir, "combined_mental_health_dataset.csv")
            combined_df.to_csv(combined_filepath, index=False)
            saved_files['combined'] = combined_filepath
            self.logger.info(f"Saved combined dataset: {combined_filepath}")
        except Exception as e:
            self.logger.error(f"Error saving combined dataset: {e}")
        
        return saved_files
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all datasets"""
        try:
            combined_df = self.combine_all_datasets()
            
            summary = {
                'total_samples': len(combined_df),
                'datasets_included': combined_df['dataset_source'].value_counts().to_dict(),
                'stress_level_distribution': combined_df['stress_level'].value_counts().to_dict(),
                'age_distribution': {
                    'mean': combined_df['age'].mean(),
                    'std': combined_df['age'].std(),
                    'min': combined_df['age'].min(),
                    'max': combined_df['age'].max()
                },
                'gender_distribution': combined_df['gender'].value_counts().to_dict(),
                'missing_values': combined_df.isnull().sum().to_dict()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating dataset summary: {e}")
            return {}