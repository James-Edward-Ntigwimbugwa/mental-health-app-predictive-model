import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic training data for stress prediction
np.random.seed(42)
n_samples = 1000

# Features: sleep_hours, work_hours, exercise_minutes, social_interactions, caffeine_intake, 
# screen_time_hours, meditation_minutes, workload_rating, relationship_satisfaction, financial_stress
X = np.random.rand(n_samples, 10)

# Scale features to realistic ranges
X[:, 0] = X[:, 0] * 5 + 4    # sleep_hours: 4-9 hours
X[:, 1] = X[:, 1] * 8 + 4    # work_hours: 4-12 hours  
X[:, 2] = X[:, 2] * 120      # exercise_minutes: 0-120 minutes
X[:, 3] = X[:, 3] * 10       # social_interactions: 0-10 per day
X[:, 4] = X[:, 4] * 5        # caffeine_intake: 0-5 cups
X[:, 5] = X[:, 5] * 12       # screen_time_hours: 0-12 hours
X[:, 6] = X[:, 6] * 60       # meditation_minutes: 0-60 minutes
X[:, 7] = X[:, 7] * 10       # workload_rating: 0-10 scale
X[:, 8] = X[:, 8] * 10       # relationship_satisfaction: 0-10 scale
X[:, 9] = X[:, 9] * 10       # financial_stress: 0-10 scale

# Create stress levels based on logical patterns
stress_scores = (
    (9 - X[:, 0]) * 0.3 +           # Less sleep = more stress
    (X[:, 1] - 6) * 0.2 +           # More work hours = more stress
    (60 - X[:, 2]) * 0.1 +          # Less exercise = more stress
    (5 - X[:, 3]) * 0.1 +           # Fewer social interactions = more stress
    X[:, 4] * 0.15 +                # More caffeine = more stress
    X[:, 5] * 0.1 +                 # More screen time = more stress
    (30 - X[:, 6]) * 0.05 +         # Less meditation = more stress
    X[:, 7] * 0.2 +                 # Higher workload = more stress
    (10 - X[:, 8]) * 0.15 +         # Lower relationship satisfaction = more stress
    X[:, 9] * 0.25                  # Higher financial stress = more stress
)

# Add some noise and normalize
stress_scores += np.random.normal(0, 1, n_samples)
stress_scores = np.clip(stress_scores, 0, 10)

# Convert to categorical stress levels
y = np.where(stress_scores < 3, 'Low', 
             np.where(stress_scores < 6, 'Moderate', 
                     np.where(stress_scores < 8, 'High', 'Severe')))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Create feature names
feature_names = [
    'sleep_hours', 'work_hours', 'exercise_minutes', 'social_interactions',
    'caffeine_intake', 'screen_time_hours', 'meditation_minutes', 
    'workload_rating', 'relationship_satisfaction', 'financial_stress'
]

# Solution recommendations database
solutions_db = {
    'Low': {
        'message': 'Your stress levels are well-managed! Keep up the good work.',
        'recommendations': [
            'Continue your current healthy habits',
            'Consider sharing your stress management techniques with others',
            'Maintain regular check-ins with yourself to monitor stress levels'
        ],
        'priority': 'maintenance'
    },
    'Moderate': {
        'message': 'You have manageable stress levels, but there\'s room for improvement.',
        'recommendations': [
            'Establish a consistent sleep schedule (7-9 hours)',
            'Incorporate 30 minutes of daily exercise',
            'Practice deep breathing exercises for 10 minutes daily',
            'Limit caffeine intake, especially in the afternoon',
            'Schedule regular social activities with friends/family',
            'Consider time management techniques to reduce workload pressure'
        ],
        'priority': 'improvement'
    },
    'High': {
        'message': 'Your stress levels are elevated and need attention.',
        'recommendations': [
            'Prioritize 7-9 hours of quality sleep nightly',
            'Implement stress-reduction techniques like meditation or yoga',
            'Consider reducing work hours or delegating tasks if possible',
            'Limit screen time, especially before bedtime',
            'Engage in regular physical activity (even 20-minute walks help)',
            'Seek support from friends, family, or support groups',
            'Practice mindfulness and relaxation techniques',
            'Consider professional counseling if stress persists'
        ],
        'priority': 'urgent'
    },
    'Severe': {
        'message': 'Your stress levels are critically high. Immediate action is recommended.',
        'recommendations': [
            'URGENT: Consider speaking with a healthcare professional or therapist',
            'Implement immediate stress relief: deep breathing, progressive muscle relaxation',
            'Prioritize sleep - aim for 8+ hours with good sleep hygiene',
            'Reduce workload immediately - delegate, postpone non-essential tasks',
            'Eliminate or significantly reduce caffeine and alcohol',
            'Engage in daily physical activity, even light walking',
            'Practice meditation or mindfulness for at least 15 minutes daily',
            'Connect with supportive friends and family members',
            'Consider stress management workshops or courses',
            'If experiencing thoughts of self-harm, contact emergency services immediately'
        ],
        'priority': 'critical'
    }
}

# Create personalized solution generator
def generate_personalized_solutions(prediction, feature_values):
    """Generate personalized solutions based on prediction and input features"""
    base_solutions = solutions_db[prediction]['recommendations'].copy()
    personalized = []
    
    # Analyze specific issues based on feature values
    sleep_hours, work_hours, exercise_min, social_int, caffeine, screen_time, meditation, workload, relationship, financial = feature_values
    
    if sleep_hours < 6:
        personalized.append("PRIORITY: Increase sleep to at least 7 hours - your current sleep is critically low")
    elif sleep_hours < 7:
        personalized.append("Focus on getting 7-8 hours of sleep instead of your current amount")
    
    if work_hours > 10:
        personalized.append("Consider reducing work hours - working 10+ hours daily significantly increases stress")
    
    if exercise_min < 30:
        personalized.append("Increase daily exercise - aim for at least 30 minutes of physical activity")
    
    if caffeine > 3:
        personalized.append("Reduce caffeine intake - you're consuming more than the recommended amount")
    
    if screen_time > 8:
        personalized.append("Limit screen time to reduce eye strain and improve sleep quality")
    
    if meditation < 10:
        personalized.append("Start with 10-15 minutes of daily meditation or mindfulness practice")
    
    if workload > 7:
        personalized.append("Address high workload through time management, delegation, or discussing with supervisor")
    
    if relationship < 5:
        personalized.append("Focus on improving relationships - consider couples counseling or communication workshops")
    
    if financial > 7:
        personalized.append("Address financial stress through budgeting, financial counseling, or debt management")
    
    return personalized + base_solutions

# Model performance metrics (calculated on test set)
test_accuracy = model.score(X_test_scaled, y_test)
feature_importance = dict(zip(feature_names, model.feature_importances_))

# Create the complete model package
stress_model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names,
    'solutions_database': solutions_db,
    'solution_generator': generate_personalized_solutions,
    'model_info': {
        'model_type': 'RandomForestClassifier',
        'training_date': datetime.now().isoformat(),
        'test_accuracy': test_accuracy,
        'feature_importance': feature_importance,
        'stress_levels': ['Low', 'Moderate', 'High', 'Severe'],
        'version': '1.0'
    },
    'usage_instructions': {
        'input_format': 'List or array of 10 numeric values in this order: ' + ', '.join(feature_names),
        'prediction_method': 'Use model.predict() on scaled input',
        'solution_method': 'Call solution_generator(prediction, input_features)'
    }
}

# Save the model package to pickle file
with open('stress_model.pkl', 'wb') as f:
    pickle.dump(stress_model_package, f)

print("Stress model package saved to 'stress_model.pkl'")
print(f"Model accuracy on test set: {test_accuracy:.3f}")
print(f"Feature importance ranking:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.3f}")

# Example usage demonstration
print("\n" + "="*50)
print("EXAMPLE USAGE:")
print("="*50)

# Load the model (demonstration)
with open('stress_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Example prediction
example_input = [6.5, 9, 30, 3, 2, 6, 15, 8, 6, 7]  # Example stress factors
example_scaled = loaded_model['scaler'].transform([example_input])
prediction = loaded_model['model'].predict(example_scaled)[0]
probability = loaded_model['model'].predict_proba(example_scaled)[0]

print(f"\nExample Input: {dict(zip(feature_names, example_input))}")
print(f"Predicted Stress Level: {prediction}")
print(f"Prediction Confidence: {max(probability):.3f}")

# Generate solutions
personalized_solutions = loaded_model['solution_generator'](prediction, example_input)
print(f"\nPersonalized Recommendations:")
for i, solution in enumerate(personalized_solutions[:5], 1):  # Show first 5
    print(f"{i}. {solution}")

print(f"\nStress level message: {loaded_model['solutions_database'][prediction]['message']}")
print(f"Priority level: {loaded_model['solutions_database'][prediction]['priority'].upper()}")