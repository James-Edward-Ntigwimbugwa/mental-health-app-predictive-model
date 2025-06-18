# src/features/problem_analyzer.py
"""
Stress Problem Analyzer - Identifies health risks and problems associated with stress levels
"""

from typing import Dict, List, Any
import logging

class StressProblemAnalyzer:
    """Analyzes problems and health risks associated with different stress levels"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stress_problems = self._initialize_stress_problems()
    
    def _initialize_stress_problems(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive stress-related problems database"""
        return {
            'Low': {
                'physical_symptoms': [
                    'Occasional headaches',
                    'Minor muscle tension',
                    'Slightly elevated heart rate during challenges'
                ],
                'mental_symptoms': [
                    'Mild worry about upcoming events',
                    'Occasional difficulty concentrating',
                    'Slight mood fluctuations'
                ],
                'behavioral_symptoms': [
                    'Minor changes in sleep patterns',
                    'Occasional procrastination',
                    'Slight changes in appetite'
                ],
                'health_risks': [
                    'Generally minimal health impact',
                    'May experience minor immune system variations',
                    'Possible slight digestive sensitivity'
                ],
                'long_term_effects': [
                    'Usually manageable without intervention',
                    'May benefit from stress management techniques',
                    'Low risk of chronic health issues'
                ],
                'severity_score': 2
            },
            
            'Moderate': {
                'physical_symptoms': [
                    'Regular headaches or migraines',
                    'Muscle tension and stiffness',
                    'Fatigue and low energy',
                    'Sleep disturbances',
                    'Digestive issues (upset stomach, acid reflux)'
                ],
                'mental_symptoms': [
                    'Anxiety and worry',
                    'Difficulty concentrating',
                    'Mood swings',
                    'Irritability',
                    'Feeling overwhelmed'
                ],
                'behavioral_symptoms': [
                    'Changes in eating habits',
                    'Increased caffeine or alcohol consumption',
                    'Social withdrawal',
                    'Procrastination',
                    'Restlessness'
                ],
                'health_risks': [
                    'Weakened immune system',
                    'Increased susceptibility to infections',
                    'Risk of developing anxiety disorders',
                    'Potential cardiovascular strain',
                    'Digestive system dysfunction'
                ],
                'long_term_effects': [
                    'Risk of chronic stress-related conditions',
                    'Potential development of depression',
                    'Increased risk of lifestyle-related diseases',
                    'Relationship and work performance issues'
                ],
                'severity_score': 5
            },
            
            'High': {
                'physical_symptoms': [
                    'Chronic headaches and migraines',
                    'Severe muscle tension and pain',
                    'Chest pain and heart palpitations',
                    'High blood pressure',
                    'Chronic fatigue and exhaustion',
                    'Frequent illness due to compromised immunity',
                    'Gastrointestinal problems (IBS, ulcers)',
                    'Skin conditions (eczema, psoriasis flare-ups)'
                ],
                'mental_symptoms': [
                    'Persistent anxiety and panic attacks',
                    'Depression and mood disorders',
                    'Cognitive impairment and memory problems',
                    'Difficulty making decisions',
                    'Feeling of helplessness',
                    'Emotional numbness or hypersensitivity'
                ],
                'behavioral_symptoms': [
                    'Significant changes in appetite (over or under-eating)',
                    'Substance abuse (alcohol, drugs, prescription medications)',
                    'Social isolation and relationship problems',
                    'Aggressive behavior or outbursts',
                    'Compulsive behaviors',
                    'Neglecting responsibilities'
                ],
                'health_risks': [
                    'Cardiovascular disease (heart attack, stroke risk)',
                    'Type 2 diabetes',
                    'Autoimmune disorders',
                    'Chronic inflammatory conditions',
                    'Mental health disorders (anxiety, depression)',
                    'Substance use disorders',
                    'Reproductive health issues',
                    'Accelerated aging'
                ],
                'long_term_effects': [
                    'Significant risk of chronic diseases',
                    'Reduced life expectancy',
                    'Severe impact on quality of life',
                    'Relationship breakdown and social isolation',
                    'Career and financial consequences',
                    'Increased healthcare costs and medical interventions'
                ],
                'severity_score': 8
            },
            
            'Severe': {
                'physical_symptoms': [
                    'Debilitating chronic pain',
                    'Severe cardiovascular symptoms',
                    'Complete immune system breakdown',
                    'Severe digestive disorders',
                    'Chronic insomnia or hypersomnia',
                    'Severe headaches and neurological symptoms',
                    'Autoimmune flare-ups',
                    'Physical collapse and hospitalization risk'
                ],
                'mental_symptoms': [
                    'Severe depression with suicidal ideation',
                    'Panic disorder and agoraphobia',
                    'Cognitive dysfunction and memory loss',
                    'Dissociative episodes',
                    'Complete emotional breakdown',
                    'Psychotic episodes in extreme cases',
                    'Severe anxiety disorders',
                    'Post-traumatic stress symptoms'
                ],
                'behavioral_symptoms': [
                    'Complete social withdrawal and isolation',
                    'Severe substance abuse and addiction',
                    'Self-harm behaviors',
                    'Complete neglect of personal care',
                    'Inability to function in daily activities',
                    'Aggressive or violent behavior',
                    'Eating disorders',
                    'Complete work or academic dysfunction'
                ],
                'health_risks': [
                    'Life-threatening cardiovascular events',
                    'Severe autoimmune conditions',
                    'Organ failure risk',
                    'Severe mental health crises',
                    'Suicide risk',
                    'Complete immune system failure',
                    'Severe metabolic disorders',
                    'Neurological damage'
                ],
                'long_term_effects': [
                    'Permanent health damage',
                    'Significantly reduced life expectancy',
                    'Complete life dysfunction',
                    'Permanent disability risk',
                    'Severe social and economic consequences',
                    'Requires immediate medical intervention',
                    'Long-term care and support needs',
                    'Generational impact on family'
                ],
                'severity_score': 10
            }
        }
    
    def analyze_problems(self, stress_level: str, user_factors: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze problems associated with the predicted stress level"""
        if stress_level not in self.stress_problems:
            stress_level = 'Moderate'  # Default fallback
        
        problems = self.stress_problems[stress_level].copy()
        
        # Personalize based on user factors if provided
        if user_factors:
            problems = self._personalize_problems(problems, user_factors, stress_level)
        
        return {
            'stress_level': stress_level,
            'severity_score': problems['severity_score'],
            'immediate_concerns': self._identify_immediate_concerns(problems),
            'physical_impact': problems['physical_symptoms'],
            'mental_impact': problems['mental_symptoms'],
            'behavioral_impact': problems['behavioral_symptoms'],
            'health_risks': problems['health_risks'],
            'long_term_consequences': problems['long_term_effects'],
            'urgency_level': self._determine_urgency(stress_level),
            'recommended_actions': self._get_immediate_actions(stress_level)
        }
    
    def _personalize_problems(self, problems: Dict[str, Any], user_factors: Dict[str, Any], stress_level: str) -> Dict[str, Any]:
        """Personalize problems based on user-specific factors"""
        # Age-based considerations
        age = user_factors.get('age', 30)
        if age > 50:
            problems['health_risks'].extend([
                'Increased risk of age-related cardiovascular issues',
                'Higher susceptibility to chronic conditions'
            ])
        elif age < 25:
            problems['mental_symptoms'].extend([
                'Academic or career development stress',
                'Identity and future uncertainty concerns'
            ])
        
        # Work-related factors
        work_hours = user_factors.get('work_hours', 8)
        if work_hours > 10:
            problems['physical_symptoms'].extend([
                'Severe work-related burnout symptoms',
                'Chronic fatigue from overwork'
            ])
        
        # Health history considerations
        if user_factors.get('chronic_illness', False):
            problems['health_risks'].extend([
                'Exacerbation of existing chronic conditions',
                'Increased medical complications'
            ])
        
        return problems
    
    def _identify_immediate_concerns(self, problems: Dict[str, Any]) -> List[str]:
        """Identify the most immediate concerns requiring attention"""
        severity = problems['severity_score']
        
        if severity >= 8:
            return [
                'Immediate medical evaluation recommended',
                'Mental health crisis intervention may be needed',
                'Risk of severe health complications',
                'Emergency support system activation advised'
            ]
        elif severity >= 5:
            return [
                'Professional healthcare consultation recommended',
                'Mental health support advised',
                'Lifestyle modifications urgently needed',
                'Monitor for worsening symptoms'
            ]
        else:
            return [
                'Preventive measures recommended',
                'Stress management techniques beneficial',
                'Regular monitoring advised'
            ]
    
    def _determine_urgency(self, stress_level: str) -> str:
        """Determine urgency level for intervention"""
        urgency_map = {
            'Low': 'Routine monitoring',
            'Moderate': 'Proactive intervention',
            'High': 'Urgent intervention required',
            'Severe': 'Emergency intervention needed'
        }
        return urgency_map.get(stress_level, 'Proactive intervention')
    
    def _get_immediate_actions(self, stress_level: str) -> List[str]:
        """Get immediate actions based on stress level"""
        actions_map = {
            'Low': [
                'Continue current stress management practices',
                'Maintain healthy lifestyle habits',
                'Consider preventive stress reduction techniques'
            ],
            'Moderate': [
                'Implement structured stress management routine',
                'Consider professional counseling or therapy',
                'Improve sleep hygiene and exercise routine',
                'Evaluate and modify stressful life circumstances'
            ],
            'High': [
                'Seek immediate professional medical evaluation',
                'Contact mental health professional',
                'Implement emergency stress reduction measures',
                'Consider medication evaluation if appropriate',
                'Activate support network immediately'
            ],
            'Severe': [
                'Seek emergency medical attention',
                'Contact crisis intervention services',
                'Immediate psychiatric evaluation',
                'Consider inpatient treatment options',
                'Emergency contact to family/support system',
                'Remove access to harmful substances/objects if applicable'
            ]
        }
        return actions_map.get(stress_level, actions_map['Moderate'])
    
    def generate_problem_report(self, stress_level: str, user_factors: Dict[str, Any] = None) -> str:
        """Generate a comprehensive problem analysis report"""
        analysis = self.analyze_problems(stress_level, user_factors)
        
        report = f"""
STRESS LEVEL ANALYSIS REPORT
============================

Current Stress Level: {analysis['stress_level']}
Severity Score: {analysis['severity_score']}/10
Urgency Level: {analysis['urgency_level']}

IMMEDIATE CONCERNS:
{chr(10).join(['• ' + concern for concern in analysis['immediate_concerns']])}

PHYSICAL IMPACT:
{chr(10).join(['• ' + symptom for symptom in analysis['physical_impact']])}

MENTAL/EMOTIONAL IMPACT:
{chr(10).join(['• ' + symptom for symptom in analysis['mental_impact']])}

BEHAVIORAL CHANGES:
{chr(10).join(['• ' + symptom for symptom in analysis['behavioral_impact']])}

HEALTH RISKS:
{chr(10).join(['• ' + risk for risk in analysis['health_risks']])}

LONG-TERM CONSEQUENCES:
{chr(10).join(['• ' + consequence for consequence in analysis['long_term_consequences']])}

IMMEDIATE RECOMMENDED ACTIONS:
{chr(10).join(['• ' + action for action in analysis['recommended_actions']])}

============================
This analysis is for informational purposes only and should not replace professional medical advice.
If you are experiencing severe symptoms, please seek immediate professional help.
"""
        return report