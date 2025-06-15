"""
QuantAI Hospital Dataset Generator
This module generates comprehensive and realistic healthcare data for training and testing the calling agent system.
Includes advanced hospital operations, patient care scenarios, and medical workflows.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
import json
from typing import List, Dict, Any
import os

class QuantAIHospitalDatasetGenerator:
    def __init__(self):
        self.fake = Faker()
        self.hospital_name = "QuantAI Hospital"
        self.owner = "Harjas Singh"
        
        # Load medical knowledge base
        self.medical_knowledge = self._load_medical_knowledge()
        
        # Initialize hospital infrastructure
        self.hospital_infrastructure = self._initialize_hospital_infrastructure()
        
        # Initialize hospital quality metrics
        self.quality_metrics = self._initialize_quality_metrics()
        
        # Initialize hospital accreditation status
        self.accreditation = self._initialize_accreditation()

    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize hospital quality metrics and performance indicators."""
        return {
            'patient_safety': {
                'infection_rates': {
                    'hais': 0.02,  # Hospital Acquired Infections
                    'surgical_site': 0.015,
                    'uti': 0.01,
                    'pneumonia': 0.008
                },
                'fall_rates': 0.03,
                'pressure_ulcer_rates': 0.02,
                'medication_errors': 0.01
            },
            'clinical_outcomes': {
                'mortality_rate': 0.02,
                'readmission_rate': 0.15,
                'complication_rate': 0.08,
                'surgical_success_rate': 0.95
            },
            'patient_experience': {
                'satisfaction_score': 4.5,
                'wait_times': {
                    'emergency': 15,  # minutes
                    'outpatient': 20,
                    'surgery': 30
                },
                'communication_score': 4.6
            },
            'operational_metrics': {
                'bed_occupancy_rate': 0.85,
                'average_length_of_stay': 4.5,  # days
                'turnaround_time': {
                    'lab_results': 45,  # minutes
                    'imaging': 60,
                    'pharmacy': 30
                }
            }
        }
        
    def _initialize_accreditation(self) -> Dict[str, Any]:
        """Initialize hospital accreditation and certification status."""
        return {
            'joint_commission': {
                'status': 'Accredited',
                'last_survey': '2023-06-15',
                'next_survey': '2024-06-15',
                'specialty_certifications': [
                    'Stroke Care',
                    'Heart Failure',
                    'Joint Replacement',
                    'Spine Surgery'
                ]
            },
            'specialty_certifications': {
                'trauma_center': 'Level I',
                'stroke_center': 'Comprehensive',
                'cancer_center': 'Comprehensive',
                'cardiac_center': 'Advanced'
            },
            'quality_recognitions': [
                'Magnet Recognition for Nursing Excellence',
                'Leapfrog Top Hospital',
                'Healthgrades Distinguished Hospital'
            ]
        }
        
    def _initialize_hospital_infrastructure(self) -> Dict[str, Any]:
        """Initialize hospital infrastructure details."""
        return {
            'departments': {
                'Cardiology': {
                    'specialties': ['Heart disease', 'Hypertension', 'Heart failure', 'Arrhythmia', 'Cardiac imaging'],
                    'common_procedures': ['EKG', 'Echocardiogram', 'Stress test', 'Cardiac catheterization', 'Angioplasty'],
                    'doctors': ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis'],
                    'nurses': ['Nurse Brown', 'Nurse Davis', 'Nurse Wilson', 'Nurse Taylor', 'Nurse Anderson'],
                    'equipment': ['ECG Machine', 'Echocardiogram', 'Stress Test Equipment', 'Cardiac Monitor'],
                    'beds': 20,
                    'operating_rooms': 2,
                    'wait_time': '15-30 minutes'
                },
                'Neurology': {
                    'specialties': ['Brain disorders', 'Nerve disorders', 'Stroke', 'Epilepsy', 'Movement disorders'],
                    'common_procedures': ['MRI', 'CT scan', 'EEG', 'Nerve conduction study', 'Lumbar puncture'],
                    'doctors': ['Dr. Brown', 'Dr. Davis', 'Dr. Miller', 'Dr. Wilson', 'Dr. Moore'],
                    'nurses': ['Nurse Smith', 'Nurse Johnson', 'Nurse Williams', 'Nurse Brown', 'Nurse Davis'],
                    'equipment': ['MRI Machine', 'CT Scanner', 'EEG Machine', 'Nerve Conduction Equipment'],
                    'beds': 15,
                    'operating_rooms': 2,
                    'wait_time': '20-40 minutes'
                },
                'Orthopedics': {
                    'specialties': ['Bone disorders', 'Joint problems', 'Sports injuries', 'Spine conditions', 'Trauma'],
                    'common_procedures': ['X-ray', 'Physical therapy', 'Joint replacement', 'Arthroscopy', 'Fracture repair'],
                    'doctors': ['Dr. Wilson', 'Dr. Moore', 'Dr. Taylor', 'Dr. Anderson', 'Dr. Thomas'],
                    'nurses': ['Nurse Brown', 'Nurse Davis', 'Nurse Wilson', 'Nurse Taylor', 'Nurse Anderson'],
                    'equipment': ['X-Ray Machine', 'Physical Therapy Equipment', 'Surgical Navigation System'],
                    'beds': 25,
                    'operating_rooms': 2,
                    'wait_time': '10-25 minutes'
                },
                'Emergency': {
                    'specialties': ['Trauma', 'Acute care', 'Critical care', 'Toxicology', 'Disaster medicine'],
                    'common_procedures': ['Trauma assessment', 'Emergency surgery', 'Critical care', 'Intubation'],
                    'doctors': ['Dr. Anderson', 'Dr. Thomas', 'Dr. Jackson', 'Dr. White', 'Dr. Harris'],
                    'nurses': ['Nurse Martinez', 'Nurse Robinson', 'Nurse Clark', 'Nurse Lewis', 'Nurse Lee'],
                    'equipment': ['Trauma Bay Equipment', 'Emergency Response Equipment', 'Critical Care Equipment'],
                    'beds': 50,
                    'operating_rooms': 2,
                    'wait_time': 'Immediate'
                },
                'Pediatrics': {
                    'specialties': ['Child health', 'Development', 'Pediatric diseases', 'Adolescent medicine', 'Neonatology'],
                    'common_procedures': ['Well-child visits', 'Vaccinations', 'Development assessment', 'Growth monitoring'],
                    'doctors': ['Dr. White', 'Dr. Harris', 'Dr. Martin', 'Dr. Thompson', 'Dr. Garcia'],
                    'nurses': ['Nurse Thompson', 'Nurse Garcia', 'Nurse Martinez', 'Nurse Robinson', 'Nurse Clark'],
                    'equipment': ['Pediatric Examination Equipment', 'Vaccination Supplies', 'Growth Charts'],
                    'beds': 30,
                    'operating_rooms': 1,
                    'wait_time': '15-30 minutes'
                }
            },
            'facilities': {
                'emergency_room': {
                    'capacity': 50,
                    'beds': {
                        'trauma': 10,
                        'acute': 20,
                        'observation': 20
                    },
                    'equipment': ['CT Scanner', 'MRI', 'X-Ray', 'Ultrasound', 'ECG'],
                    'specialized_units': {
                        'trauma_center': True,
                        'stroke_center': True,
                        'cardiac_center': True
                    }
                },
                'operating_rooms': {
                    'total': 8,
                    'specialized': {
                        'cardiac': 2,
                        'neurology': 2,
                        'orthopedics': 2,
                        'general': 2
                    },
                    'equipment': {
                        'anesthesia_machines': 8,
                        'surgical_lights': 16,
                        'monitoring_systems': 8,
                        'ventilators': 8
                    }
                },
                'icu': {
                    'medical': 15,
                    'surgical': 10,
                    'cardiac': 10,
                    'neurological': 10,
                    'pediatric': 5,
                    'equipment': {
                        'ventilators': 30,
                        'monitoring_systems': 50,
                        'dialysis_machines': 5
                    }
                },
                'wards': {
                    'general': 100,
                    'private': 50,
                    'semi_private': 75,
                    'pediatric': 30,
                    'maternity': 40,
                    'oncology': 25,
                    'isolation': 10
                }
            }
        }
        
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base with conditions, symptoms, and treatments."""
        return {
            'conditions': {
                'Hypertension': {
                    'symptoms': ['High blood pressure', 'Headaches', 'Dizziness', 'Chest pain', 'Shortness of breath', 'Nosebleeds'],
                    'risk_factors': ['Age', 'Family history', 'Obesity', 'Smoking', 'High salt diet', 'Stress', 'Alcohol consumption'],
                    'treatments': ['Lifestyle changes', 'Medication', 'Regular monitoring', 'Diet modification', 'Exercise program'],
                    'medications': ['Lisinopril', 'Amlodipine', 'Metoprolol', 'Hydrochlorothiazide', 'Losartan', 'Valsartan'],
                    'icd10_code': 'I10',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Crisis'],
                    'follow_up_frequency': '3 months',
                    'complications': ['Heart disease', 'Stroke', 'Kidney damage', 'Vision problems'],
                    'preventive_measures': ['Regular exercise', 'Healthy diet', 'Stress management', 'Regular check-ups']
                },
                'Type 2 Diabetes': {
                    'symptoms': ['Increased thirst', 'Frequent urination', 'Fatigue', 'Blurred vision', 'Slow healing', 'Weight loss', 'Numbness'],
                    'risk_factors': ['Family history', 'Obesity', 'Physical inactivity', 'Age', 'Race', 'Gestational diabetes'],
                    'treatments': ['Insulin therapy', 'Diet control', 'Exercise', 'Blood sugar monitoring', 'Medication management'],
                    'medications': ['Metformin', 'Insulin', 'Glipizide', 'Sitagliptin', 'Empagliflozin', 'Canagliflozin'],
                    'icd10_code': 'E11',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Advanced'],
                    'follow_up_frequency': '3 months',
                    'complications': ['Heart disease', 'Kidney damage', 'Nerve damage', 'Eye damage', 'Foot problems'],
                    'preventive_measures': ['Healthy diet', 'Regular exercise', 'Weight management', 'Regular screening']
                },
                'Asthma': {
                    'symptoms': ['Wheezing', 'Shortness of breath', 'Chest tightness', 'Coughing', 'Difficulty breathing', 'Sleep problems'],
                    'risk_factors': ['Allergies', 'Family history', 'Environmental factors', 'Respiratory infections', 'Exercise'],
                    'treatments': ['Inhalers', 'Avoiding triggers', 'Regular check-ups', 'Action plan', 'Peak flow monitoring'],
                    'medications': ['Albuterol', 'Fluticasone', 'Montelukast', 'Salmeterol', 'Budesonide', 'Formoterol'],
                    'icd10_code': 'J45',
                    'severity_levels': ['Intermittent', 'Mild Persistent', 'Moderate Persistent', 'Severe Persistent'],
                    'follow_up_frequency': '6 months',
                    'complications': ['Severe attacks', 'Sleep problems', 'Reduced activity', 'Emergency visits'],
                    'preventive_measures': ['Trigger avoidance', 'Regular medication', 'Action plan', 'Regular monitoring']
                },
                'Anxiety': {
                    'symptoms': ['Excessive worry', 'Restlessness', 'Fatigue', 'Difficulty concentrating', 'Irritability', 'Sleep problems'],
                    'risk_factors': ['Family history', 'Trauma', 'Stress', 'Medical conditions', 'Substance abuse'],
                    'treatments': ['Therapy', 'Medication', 'Lifestyle changes', 'Stress management', 'Support groups'],
                    'medications': ['Sertraline', 'Fluoxetine', 'Escitalopram', 'Buspirone', 'Alprazolam', 'Diazepam'],
                    'icd10_code': 'F41',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Panic'],
                    'follow_up_frequency': '1 month',
                    'complications': ['Depression', 'Substance abuse', 'Social isolation', 'Physical health problems'],
                    'preventive_measures': ['Stress management', 'Regular therapy', 'Social support', 'Healthy lifestyle']
                },
                'Depression': {
                    'symptoms': ['Persistent sadness', 'Loss of interest', 'Fatigue', 'Sleep changes', 'Appetite changes', 'Concentration problems'],
                    'risk_factors': ['Family history', 'Trauma', 'Stress', 'Medical conditions', 'Substance abuse'],
                    'treatments': ['Therapy', 'Medication', 'Lifestyle changes', 'Support groups', 'Exercise'],
                    'medications': ['Sertraline', 'Fluoxetine', 'Escitalopram', 'Bupropion', 'Venlafaxine', 'Mirtazapine'],
                    'icd10_code': 'F32',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Major'],
                    'follow_up_frequency': '1 month',
                    'complications': ['Suicide risk', 'Substance abuse', 'Social isolation', 'Physical health problems'],
                    'preventive_measures': ['Stress management', 'Regular therapy', 'Social support', 'Healthy lifestyle']
                },
                'Migraine': {
                    'symptoms': ['Severe headache', 'Nausea', 'Vomiting', 'Sensitivity to light', 'Sensitivity to sound', 'Aura'],
                    'risk_factors': ['Family history', 'Age', 'Gender', 'Hormonal changes', 'Stress', 'Certain foods'],
                    'treatments': ['Pain medication', 'Preventive medication', 'Lifestyle changes', 'Trigger avoidance', 'Stress management'],
                    'medications': ['Sumatriptan', 'Rizatriptan', 'Propranolol', 'Topiramate', 'Amitriptyline', 'Botox'],
                    'icd10_code': 'G43',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Chronic'],
                    'follow_up_frequency': '3 months',
                    'complications': ['Status migrainosus', 'Chronic migraine', 'Medication overuse', 'Depression'],
                    'preventive_measures': ['Trigger identification', 'Regular sleep', 'Stress management', 'Regular exercise']
                },
                'Gastroesophageal Reflux Disease': {
                    'symptoms': ['Heartburn', 'Regurgitation', 'Chest pain', 'Difficulty swallowing', 'Chronic cough', 'Hoarseness'],
                    'risk_factors': ['Obesity', 'Smoking', 'Pregnancy', 'Hiatal hernia', 'Certain medications', 'Diet'],
                    'treatments': ['Lifestyle changes', 'Medication', 'Diet modification', 'Weight management', 'Surgery'],
                    'medications': ['Omeprazole', 'Esomeprazole', 'Pantoprazole', 'Ranitidine', 'Famotidine', 'Antacids'],
                    'icd10_code': 'K21',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Complicated'],
                    'follow_up_frequency': '6 months',
                    'complications': ['Esophagitis', 'Barrett\'s esophagus', 'Esophageal stricture', 'Esophageal cancer'],
                    'preventive_measures': ['Diet modification', 'Weight management', 'Smoking cessation', 'Elevated sleeping position']
                },
                'Osteoarthritis': {
                    'symptoms': ['Joint pain', 'Stiffness', 'Swelling', 'Reduced range of motion', 'Grating sensation', 'Bone spurs'],
                    'risk_factors': ['Age', 'Obesity', 'Joint injury', 'Overuse', 'Genetics', 'Gender'],
                    'treatments': ['Physical therapy', 'Pain medication', 'Lifestyle changes', 'Weight management', 'Surgery'],
                    'medications': ['Ibuprofen', 'Naproxen', 'Acetaminophen', 'Diclofenac', 'Celecoxib', 'Hyaluronic acid'],
                    'icd10_code': 'M17',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Advanced'],
                    'follow_up_frequency': '6 months',
                    'complications': ['Joint deformity', 'Reduced mobility', 'Chronic pain', 'Disability'],
                    'preventive_measures': ['Weight management', 'Regular exercise', 'Joint protection', 'Early treatment']
                },
                'Chronic Back Pain': {
                    'symptoms': ['Persistent pain', 'Stiffness', 'Muscle ache', 'Limited mobility', 'Radiating pain', 'Numbness'],
                    'risk_factors': ['Age', 'Obesity', 'Poor posture', 'Physical work', 'Smoking', 'Psychological factors'],
                    'treatments': ['Physical therapy', 'Pain medication', 'Exercise', 'Lifestyle changes', 'Surgery'],
                    'medications': ['Ibuprofen', 'Naproxen', 'Acetaminophen', 'Muscle relaxants', 'Opioids', 'Antidepressants'],
                    'icd10_code': 'M54',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Chronic'],
                    'follow_up_frequency': '3 months',
                    'complications': ['Disability', 'Depression', 'Sleep problems', 'Reduced quality of life'],
                    'preventive_measures': ['Proper posture', 'Regular exercise', 'Weight management', 'Ergonomic workspace']
                },
                'Allergic Rhinitis': {
                    'symptoms': ['Sneezing', 'Runny nose', 'Nasal congestion', 'Itchy eyes', 'Postnasal drip', 'Fatigue'],
                    'risk_factors': ['Family history', 'Allergies', 'Environmental exposure', 'Age', 'Gender', 'Other allergies'],
                    'treatments': ['Antihistamines', 'Nasal sprays', 'Allergy shots', 'Avoiding triggers', 'Decongestants'],
                    'medications': ['Cetirizine', 'Loratadine', 'Fluticasone', 'Mometasone', 'Ipratropium', 'Montelukast'],
                    'icd10_code': 'J30',
                    'severity_levels': ['Mild', 'Moderate', 'Severe', 'Persistent'],
                    'follow_up_frequency': '6 months',
                    'complications': ['Sinusitis', 'Ear infections', 'Sleep problems', 'Asthma'],
                    'preventive_measures': ['Allergen avoidance', 'Regular cleaning', 'Air filters', 'Proper ventilation']
                },
                'Glaucoma': {
                    'symptoms': ['Gradual vision loss', 'Tunnel vision', 'Eye pain', 'Headaches', 'Nausea', 'Vomiting'],
                    'risk_factors': ['Age', 'Family history', 'High eye pressure', 'Ethnicity', 'Medical conditions', 'Eye injuries'],
                    'treatments': ['Eye drops', 'Oral medications', 'Laser therapy', 'Surgery', 'Regular monitoring'],
                    'medications': ['Timolol', 'Latanoprost', 'Brimonidine', 'Dorzolamide', 'Pilocarpine', 'Acetazolamide'],
                    'icd10_code': 'H40',
                    'severity_levels': ['Early', 'Moderate', 'Advanced', 'Severe'],
                    'follow_up_frequency': '3 months',
                    'complications': ['Vision loss', 'Blindness', 'Eye pressure problems', 'Surgical complications'],
                    'preventive_measures': ['Regular eye exams', 'Early detection', 'Medication adherence', 'Protective eyewear']
                }
            },
            'departments': {
                'Cardiology': {
                    'specialties': ['Heart disease', 'Hypertension', 'Heart failure', 'Arrhythmia', 'Cardiac imaging'],
                    'common_procedures': ['EKG', 'Echocardiogram', 'Stress test', 'Cardiac catheterization', 'Angioplasty', 'Pacemaker insertion'],
                    'doctors': ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis'],
                    'nurses': ['Nurse Brown', 'Nurse Davis', 'Nurse Wilson', 'Nurse Taylor', 'Nurse Anderson'],
                    'equipment': ['ECG Machine', 'Echocardiogram', 'Stress Test Equipment', 'Cardiac Monitor', 'Defibrillator'],
                    'beds': 20,
                    'operating_rooms': 2,
                    'wait_time': '15-30 minutes',
                    'specialized_units': ['Cardiac ICU', 'Cardiac Rehabilitation', 'Electrophysiology Lab'],
                    'research_areas': ['Heart Failure', 'Cardiac Imaging', 'Preventive Cardiology']
                },
                'Neurology': {
                    'specialties': ['Brain disorders', 'Nerve disorders', 'Stroke', 'Epilepsy', 'Movement disorders'],
                    'common_procedures': ['MRI', 'CT scan', 'EEG', 'Nerve conduction study', 'Lumbar puncture', 'Botulinum toxin injection'],
                    'doctors': ['Dr. Brown', 'Dr. Davis', 'Dr. Miller', 'Dr. Wilson', 'Dr. Moore'],
                    'nurses': ['Nurse Smith', 'Nurse Johnson', 'Nurse Williams', 'Nurse Brown', 'Nurse Davis'],
                    'equipment': ['MRI Machine', 'CT Scanner', 'EEG Machine', 'Nerve Conduction Equipment', 'TMS Device'],
                    'beds': 15,
                    'operating_rooms': 2,
                    'wait_time': '20-40 minutes',
                    'specialized_units': ['Stroke Unit', 'Epilepsy Monitoring', 'Neuromuscular Clinic'],
                    'research_areas': ['Stroke Treatment', 'Neurodegenerative Diseases', 'Brain Mapping']
                },
                'Orthopedics': {
                    'specialties': ['Bone disorders', 'Joint problems', 'Sports injuries', 'Spine conditions', 'Trauma'],
                    'common_procedures': ['X-ray', 'Physical therapy', 'Joint replacement', 'Arthroscopy', 'Fracture repair', 'Spine surgery'],
                    'doctors': ['Dr. Wilson', 'Dr. Moore', 'Dr. Taylor', 'Dr. Anderson', 'Dr. Thomas'],
                    'nurses': ['Nurse Brown', 'Nurse Davis', 'Nurse Wilson', 'Nurse Taylor', 'Nurse Anderson'],
                    'equipment': ['X-Ray Machine', 'Physical Therapy Equipment', 'Surgical Navigation System', 'Arthroscopy Equipment'],
                    'beds': 25,
                    'operating_rooms': 2,
                    'wait_time': '10-25 minutes',
                    'specialized_units': ['Sports Medicine', 'Joint Replacement Center', 'Spine Center'],
                    'research_areas': ['Joint Replacement', 'Sports Medicine', 'Regenerative Medicine']
                },
                'Emergency': {
                    'specialties': ['Trauma', 'Acute care', 'Critical care', 'Toxicology', 'Disaster medicine'],
                    'common_procedures': ['Trauma assessment', 'Emergency surgery', 'Critical care', 'Intubation', 'Chest tube insertion'],
                    'doctors': ['Dr. Anderson', 'Dr. Thomas', 'Dr. Jackson', 'Dr. White', 'Dr. Harris'],
                    'nurses': ['Nurse Martinez', 'Nurse Robinson', 'Nurse Clark', 'Nurse Lewis', 'Nurse Lee'],
                    'equipment': ['Trauma Bay Equipment', 'Emergency Response Equipment', 'Critical Care Equipment', 'Imaging Equipment'],
                    'beds': 50,
                    'operating_rooms': 2,
                    'wait_time': 'Immediate',
                    'specialized_units': ['Trauma Center', 'Stroke Center', 'Chest Pain Center'],
                    'research_areas': ['Trauma Care', 'Emergency Medicine', 'Disaster Response']
                },
                'Pediatrics': {
                    'specialties': ['Child health', 'Development', 'Pediatric diseases', 'Adolescent medicine', 'Neonatology'],
                    'common_procedures': ['Well-child visits', 'Vaccinations', 'Development assessment', 'Growth monitoring', 'Behavioral assessment'],
                    'doctors': ['Dr. White', 'Dr. Harris', 'Dr. Martin', 'Dr. Thompson', 'Dr. Garcia'],
                    'nurses': ['Nurse Thompson', 'Nurse Garcia', 'Nurse Martinez', 'Nurse Robinson', 'Nurse Clark'],
                    'equipment': ['Pediatric Examination Equipment', 'Vaccination Supplies', 'Growth Charts', 'Development Assessment Tools'],
                    'beds': 30,
                    'operating_rooms': 1,
                    'wait_time': '15-30 minutes',
                    'specialized_units': ['Neonatal ICU', 'Pediatric Emergency', 'Child Development Center'],
                    'research_areas': ['Child Development', 'Pediatric Diseases', 'Vaccine Research']
                }
            },
            'insurance_providers': {
                'Blue Cross': {
                    'coverage_levels': ['Basic', 'Standard', 'Premium', 'Platinum'],
                    'network_status': 'In-network',
                    'pre_authorization_required': True,
                    'coverage_details': {
                        'preventive_care': 100,
                        'specialist_visits': 80,
                        'emergency': 100,
                        'hospitalization': 90,
                        'prescription_drugs': 80
                    },
                    'deductible': {
                        'individual': 1500,
                        'family': 3000
                    },
                    'copay': {
                        'primary_care': 25,
                        'specialist': 40,
                        'emergency': 100,
                        'urgent_care': 50
                    }
                },
                'Aetna': {
                    'coverage_levels': ['Essential', 'Enhanced', 'Elite', 'Premier'],
                    'network_status': 'In-network',
                    'pre_authorization_required': True,
                    'coverage_details': {
                        'preventive_care': 100,
                        'specialist_visits': 85,
                        'emergency': 100,
                        'hospitalization': 95,
                        'prescription_drugs': 85
                    },
                    'deductible': {
                        'individual': 2000,
                        'family': 4000
                    },
                    'copay': {
                        'primary_care': 30,
                        'specialist': 45,
                        'emergency': 150,
                        'urgent_care': 60
                    }
                },
                'Medicare': {
                    'coverage_levels': ['Part A', 'Part B', 'Part D', 'Medicare Advantage'],
                    'network_status': 'In-network',
                    'pre_authorization_required': False,
                    'coverage_details': {
                        'preventive_care': 100,
                        'specialist_visits': 80,
                        'emergency': 100,
                        'hospitalization': 100,
                        'prescription_drugs': 75
                    },
                    'deductible': {
                        'part_a': 1484,
                        'part_b': 203
                    },
                    'copay': {
                        'primary_care': 20,
                        'specialist': 20,
                        'emergency': 0,
                        'urgent_care': 20
                    }
                },
                'Medicaid': {
                    'coverage_levels': ['Standard', 'Expanded', 'Children\'s Health'],
                    'network_status': 'In-network',
                    'pre_authorization_required': False,
                    'coverage_details': {
                        'preventive_care': 100,
                        'specialist_visits': 100,
                        'emergency': 100,
                        'hospitalization': 100,
                        'prescription_drugs': 100
                    },
                    'deductible': {
                        'individual': 0,
                        'family': 0
                    },
                    'copay': {
                        'primary_care': 0,
                        'specialist': 0,
                        'emergency': 0,
                        'urgent_care': 0
                    }
                }
            }
        }
        
    def generate_patient_demographics(self, num_patients=1000):
        """Generate comprehensive patient demographic data with enhanced medical scenarios."""
        patients = []
        
        for _ in range(num_patients):
            age = self._generate_realistic_age()
            gender = random.choice(['M', 'F', 'Other'])
            
            patient = {
                'patient_id': f"QH{random.randint(100000, 999999)}",
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'date_of_birth': self.fake.date_of_birth(minimum_age=age, maximum_age=age),
                'age': age,
                'gender': gender,
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed', 'Separated']),
                'contact_info': {
                    'phone': self.fake.phone_number(),
                    'email': self.fake.email(),
                    'emergency_contact': {
                        'name': self.fake.name(),
                        'relationship': random.choice(['Spouse', 'Parent', 'Child', 'Sibling', 'Friend']),
                        'phone': self.fake.phone_number()
                    }
                },
                'address': {
                    'street': self.fake.street_address(),
                    'city': self.fake.city(),
                    'state': self.fake.state(),
                    'zip_code': self.fake.zipcode(),
                    'country': 'USA'
                },
                'insurance': self._generate_insurance_info(age),
                'socioeconomic': self._generate_socioeconomic_status(),
                'lifestyle': self._generate_lifestyle_factors(),
                'family_structure': self._generate_family_structure(),
                'vital_signs': self._generate_vital_signs(age),
                'immunization_history': self._generate_immunization_history(age),
                'social_determinants': self._generate_social_determinants(),
                'preventive_care': self._generate_preventive_care(age),
                'risk_factors': self._generate_risk_factors(),
                'cultural_preferences': self._generate_cultural_preferences(),
                'accessibility_needs': self._generate_accessibility_needs(),
                'allergies': self._generate_allergies(),
                'family_history': self._generate_family_history(),
                'hospital_specific': {
                    'preferred_language': random.choice(['English', 'Spanish', 'Mandarin', 'Hindi', 'Arabic']),
                    'religious_preferences': random.choice(['None', 'Christian', 'Muslim', 'Hindu', 'Buddhist', 'Jewish']),
                    'end_of_life_preferences': random.choice(['Full Code', 'DNR', 'DNR/DNI', 'Comfort Care']),
                    'organ_donor_status': random.choice(['Yes', 'No', 'Not Specified']),
                    'advance_directives': random.choice(['Yes', 'No', 'In Progress']),
                    'power_of_attorney': random.choice(['Yes', 'No', 'Not Specified']),
                    'living_will': random.choice(['Yes', 'No', 'Not Specified']),
                    'preferred_physician': random.choice(['Yes', 'No']),
                    'preferred_pharmacy': random.choice(['Yes', 'No']),
                    'transportation_needs': random.choice(['None', 'Wheelchair', 'Ambulance', 'Medical Transport']),
                    'special_accommodations': random.choice(['None', 'Sign Language', 'Braille', 'Large Print', 'Wheelchair Access']),
                    'financial_assistance': random.choice(['None', 'Medicaid', 'Charity Care', 'Payment Plan']),
                    'social_work_referral': random.choice(['Yes', 'No']),
                    'case_management': random.choice(['Yes', 'No']),
                    'discharge_planning': random.choice(['Yes', 'No']),
                    'home_health_needs': random.choice(['None', 'Nursing', 'Therapy', 'Medical Equipment']),
                    'palliative_care': random.choice(['Yes', 'No']),
                    'hospice_care': random.choice(['Yes', 'No']),
                    'mental_health_support': random.choice(['Yes', 'No']),
                    'substance_abuse_support': random.choice(['Yes', 'No']),
                    'nutritional_support': random.choice(['Yes', 'No']),
                    'rehabilitation_needs': random.choice(['None', 'Physical', 'Occupational', 'Speech']),
                    'durable_medical_equipment': random.choice(['None', 'Wheelchair', 'Walker', 'Hospital Bed']),
                    'home_modifications': random.choice(['None', 'Ramps', 'Grab Bars', 'Stair Lift']),
                    'caregiver_support': random.choice(['None', 'Family', 'Professional', 'Both']),
                    'support_groups': random.choice(['Yes', 'No']),
                    'community_resources': random.choice(['Yes', 'No']),
                    'telehealth_preference': random.choice(['Yes', 'No']),
                    'mobile_app_usage': random.choice(['Yes', 'No']),
                    'patient_portal_usage': random.choice(['Yes', 'No']),
                    'preferred_communication': random.choice(['Phone', 'Email', 'Text', 'Portal']),
                    'appointment_reminders': random.choice(['Phone', 'Email', 'Text', 'Portal']),
                    'medication_reminders': random.choice(['Phone', 'Email', 'Text', 'Portal']),
                    'follow_up_preference': random.choice(['In Person', 'Telehealth', 'Phone']),
                    'language_interpreter': random.choice(['Yes', 'No']),
                    'cultural_liaison': random.choice(['Yes', 'No']),
                    'spiritual_care': random.choice(['Yes', 'No']),
                    'pet_therapy': random.choice(['Yes', 'No']),
                    'music_therapy': random.choice(['Yes', 'No']),
                    'art_therapy': random.choice(['Yes', 'No']),
                    'massage_therapy': random.choice(['Yes', 'No']),
                    'acupuncture': random.choice(['Yes', 'No']),
                    'meditation': random.choice(['Yes', 'No']),
                    'yoga': random.choice(['Yes', 'No']),
                    'nutrition_counseling': random.choice(['Yes', 'No']),
                    'smoking_cessation': random.choice(['Yes', 'No']),
                    'weight_management': random.choice(['Yes', 'No']),
                    'diabetes_education': random.choice(['Yes', 'No']),
                    'cardiac_rehabilitation': random.choice(['Yes', 'No']),
                    'pulmonary_rehabilitation': random.choice(['Yes', 'No']),
                    'stroke_rehabilitation': random.choice(['Yes', 'No']),
                    'orthopedic_rehabilitation': random.choice(['Yes', 'No']),
                    'wound_care': random.choice(['Yes', 'No']),
                    'infusion_services': random.choice(['Yes', 'No']),
                    'dialysis': random.choice(['Yes', 'No']),
                    'radiation_therapy': random.choice(['Yes', 'No']),
                    'chemotherapy': random.choice(['Yes', 'No']),
                    'clinical_trials': random.choice(['Yes', 'No']),
                    'genetic_counseling': random.choice(['Yes', 'No']),
                    'fertility_services': random.choice(['Yes', 'No']),
                    'maternity_services': random.choice(['Yes', 'No']),
                    'pediatric_services': random.choice(['Yes', 'No']),
                    'geriatric_services': random.choice(['Yes', 'No']),
                    'bariatric_services': random.choice(['Yes', 'No']),
                    'transplant_services': random.choice(['Yes', 'No']),
                    'trauma_services': random.choice(['Yes', 'No']),
                    'burn_services': random.choice(['Yes', 'No']),
                    'wound_services': random.choice(['Yes', 'No']),
                    'hyperbaric_services': random.choice(['Yes', 'No']),
                    'sleep_services': random.choice(['Yes', 'No']),
                    'epilepsy_services': random.choice(['Yes', 'No']),
                    'movement_disorder_services': random.choice(['Yes', 'No']),
                    'memory_disorder_services': random.choice(['Yes', 'No']),
                    'autism_services': random.choice(['Yes', 'No']),
                    'adhd_services': random.choice(['Yes', 'No']),
                    'eating_disorder_services': random.choice(['Yes', 'No']),
                    'addiction_services': random.choice(['Yes', 'No']),
                    'pain_management': random.choice(['Yes', 'No']),
                    'palliative_care': random.choice(['Yes', 'No']),
                    'hospice_care': random.choice(['Yes', 'No']),
                    'bereavement_services': random.choice(['Yes', 'No']),
                    'grief_counseling': random.choice(['Yes', 'No']),
                    'family_counseling': random.choice(['Yes', 'No']),
                    'marriage_counseling': random.choice(['Yes', 'No']),
                    'child_counseling': random.choice(['Yes', 'No']),
                    'adolescent_counseling': random.choice(['Yes', 'No']),
                    'adult_counseling': random.choice(['Yes', 'No']),
                    'geriatric_counseling': random.choice(['Yes', 'No']),
                    'group_therapy': random.choice(['Yes', 'No']),
                    'individual_therapy': random.choice(['Yes', 'No']),
                    'family_therapy': random.choice(['Yes', 'No']),
                    'couples_therapy': random.choice(['Yes', 'No']),
                    'play_therapy': random.choice(['Yes', 'No']),
                    'art_therapy': random.choice(['Yes', 'No']),
                    'music_therapy': random.choice(['Yes', 'No']),
                    'dance_therapy': random.choice(['Yes', 'No']),
                    'drama_therapy': random.choice(['Yes', 'No']),
                    'poetry_therapy': random.choice(['Yes', 'No']),
                    'bibliotherapy': random.choice(['Yes', 'No']),
                    'pet_therapy': random.choice(['Yes', 'No']),
                    'equine_therapy': random.choice(['Yes', 'No']),
                    'wilderness_therapy': random.choice(['Yes', 'No']),
                    'adventure_therapy': random.choice(['Yes', 'No']),
                    'recreation_therapy': random.choice(['Yes', 'No']),
                    'occupational_therapy': random.choice(['Yes', 'No']),
                    'physical_therapy': random.choice(['Yes', 'No']),
                    'speech_therapy': random.choice(['Yes', 'No']),
                    'respiratory_therapy': random.choice(['Yes', 'No']),
                    'cardiac_rehabilitation': random.choice(['Yes', 'No']),
                    'pulmonary_rehabilitation': random.choice(['Yes', 'No']),
                    'stroke_rehabilitation': random.choice(['Yes', 'No']),
                    'orthopedic_rehabilitation': random.choice(['Yes', 'No']),
                    'neurological_rehabilitation': random.choice(['Yes', 'No']),
                    'vestibular_rehabilitation': random.choice(['Yes', 'No']),
                    'balance_rehabilitation': random.choice(['Yes', 'No']),
                    'gait_rehabilitation': random.choice(['Yes', 'No']),
                    'swallowing_rehabilitation': random.choice(['Yes', 'No']),
                    'cognitive_rehabilitation': random.choice(['Yes', 'No']),
                    'vision_rehabilitation': random.choice(['Yes', 'No']),
                    'hearing_rehabilitation': random.choice(['Yes', 'No']),
                    'voice_rehabilitation': random.choice(['Yes', 'No']),
                    'laryngeal_rehabilitation': random.choice(['Yes', 'No']),
                    'facial_rehabilitation': random.choice(['Yes', 'No']),
                    'hand_rehabilitation': random.choice(['Yes', 'No']),
                    'upper_extremity_rehabilitation': random.choice(['Yes', 'No']),
                    'lower_extremity_rehabilitation': random.choice(['Yes', 'No']),
                    'spine_rehabilitation': random.choice(['Yes', 'No']),
                    'pelvic_rehabilitation': random.choice(['Yes', 'No'])
                }
            }
            
            patients.append(patient)
            
        return pd.DataFrame(patients)

    def _generate_realistic_age(self) -> int:
        """Generate age with realistic distribution."""
        age_groups = {
            '0-17': 0.15,  # 15% children
            '18-30': 0.20,  # 20% young adults
            '31-50': 0.30,  # 30% middle-aged
            '51-70': 0.25,  # 25% seniors
            '71+': 0.10     # 10% elderly
        }
        
        age_group = random.choices(
            list(age_groups.keys()),
            weights=list(age_groups.values())
        )[0]
        
        if age_group == '0-17':
            return random.randint(0, 17)
        elif age_group == '18-30':
            return random.randint(18, 30)
        elif age_group == '31-50':
            return random.randint(31, 50)
        elif age_group == '51-70':
            return random.randint(51, 70)
        else:
            return random.randint(71, 90)

    def _generate_insurance_info(self, age: int) -> Dict[str, Any]:
        """Generate appropriate insurance information based on age."""
        if age < 18:
            providers = ['Medicaid', 'CHIP', 'Private Insurance']
            coverage = {'preventive_care': 100, 'specialist_visits': 80, 'emergency': 100}
        elif age >= 65:
            providers = ['Medicare', 'Medicare Advantage', 'Medigap']
            coverage = {'preventive_care': 100, 'specialist_visits': 80, 'emergency': 100}
        else:
            providers = ['Blue Cross', 'Aetna', 'United Healthcare', 'Cigna', 'Kaiser Permanente']
            coverage = {'preventive_care': 90, 'specialist_visits': 70, 'emergency': 90}
        
        provider = random.choice(providers)
        return {
            'provider': provider,
            'id': f"INS{random.randint(100000, 999999)}",
            'type': 'Primary' if random.random() < 0.8 else 'Secondary',
            'coverage': coverage
        }

    def _generate_socioeconomic_status(self) -> Dict[str, Any]:
        """Generate socioeconomic status information."""
        industries = [
            'Healthcare',
            'Technology',
            'Finance',
            'Education',
            'Manufacturing',
            'Retail',
            'Construction',
            'Transportation',
            'Agriculture',
            'Energy',
            'Media',
            'Hospitality',
            'Real Estate',
            'Government',
            'Non-Profit',
            'Telecommunications',
            'Entertainment',
            'Professional Services',
            'Insurance',
            'Legal Services'
        ]
        
        return {
            'income_level': random.choice(['Low', 'Lower-middle', 'Middle', 'Upper-middle', 'High']),
            'housing_status': random.choice(['Own', 'Rent', 'Homeless', 'Temporary']),
            'employment_industry': random.choice(industries),
            'education_attainment': random.choice(['High School', 'Associate Degree', 'Bachelor Degree', 'Master Degree', 'Doctorate']),
            'healthcare_access': {
                'primary_care': random.random() < 0.9,
                'specialist_care': random.random() < 0.7,
                'emergency_care': True,
                'preventive_care': random.random() < 0.8
            }
        }

    def _generate_lifestyle_factors(self) -> Dict[str, Any]:
        """Generate lifestyle factors information."""
        return {
            'smoking_status': random.choice(['Never', 'Former', 'Current']),
            'alcohol_consumption': random.choice(['None', 'Occasional', 'Moderate', 'Heavy']),
            'exercise_frequency': random.choice(['None', 'Rarely', 'Weekly', 'Daily']),
            'diet_type': random.choice(['Standard', 'Vegetarian', 'Vegan', 'Mediterranean', 'Low-carb', 'Gluten-free']),
            'sleep_pattern': {
                'average_hours': random.randint(4, 10),
                'quality': random.choice(['Poor', 'Fair', 'Good', 'Excellent'])
            },
            'stress_level': random.choice(['Low', 'Moderate', 'High']),
            'occupational_hazards': random.sample(['Chemical exposure', 'Physical strain', 'Mental stress', 'Irregular hours'], 
                                               random.randint(0, 3))
        }

    def _generate_family_structure(self) -> Dict[str, Any]:
        """Generate family structure information."""
        return {
            'household_size': random.randint(1, 6),
            'children': random.randint(0, 4),
            'elderly_dependents': random.randint(0, 2),
            'primary_caregiver': random.choice(['Self', 'Spouse', 'Child', 'Parent', 'Professional']),
            'family_support': random.choice(['Strong', 'Moderate', 'Limited', 'None']),
            'living_arrangement': random.choice(['Alone', 'With Family', 'With Roommates', 'Assisted Living'])
        }

    def _generate_vital_signs(self, age: int) -> Dict[str, Any]:
        """Generate realistic vital signs based on age."""
        if age < 18:
            return {
                'blood_pressure': f"{random.randint(90, 120)}/{random.randint(60, 80)}",
                'heart_rate': random.randint(60, 100),
                'respiratory_rate': random.randint(12, 20),
                'temperature': round(random.uniform(97.0, 99.0), 1),
                'height': round(random.uniform(100, 180), 1),
                'weight': round(random.uniform(20, 80), 1),
                'bmi': round(random.uniform(15, 25), 1)
            }
        else:
            return {
                'blood_pressure': f"{random.randint(110, 140)}/{random.randint(70, 90)}",
                'heart_rate': random.randint(60, 100),
                'respiratory_rate': random.randint(12, 20),
                'temperature': round(random.uniform(97.0, 99.0), 1),
                'height': round(random.uniform(150, 190), 1),
                'weight': round(random.uniform(45, 120), 1),
                'bmi': round(random.uniform(18, 35), 1)
            }

    def _generate_immunization_history(self, age: int) -> List[Dict[str, Any]]:
        """Generate immunization history based on age."""
        immunizations = []
        if age < 18:
            # Child immunizations
            child_vaccines = [
                {'name': 'MMR', 'doses': 2, 'last_dose': self.fake.date_between(start_date='-5y', end_date='today')},
                {'name': 'DTaP', 'doses': 5, 'last_dose': self.fake.date_between(start_date='-5y', end_date='today')},
                {'name': 'Polio', 'doses': 4, 'last_dose': self.fake.date_between(start_date='-5y', end_date='today')},
                {'name': 'Hepatitis B', 'doses': 3, 'last_dose': self.fake.date_between(start_date='-5y', end_date='today')}
            ]
            immunizations.extend(child_vaccines)
        else:
            # Adult immunizations
            adult_vaccines = [
                {'name': 'Tdap', 'doses': 1, 'last_dose': self.fake.date_between(start_date='-10y', end_date='today')},
                {'name': 'Influenza', 'doses': 1, 'last_dose': self.fake.date_between(start_date='-1y', end_date='today')},
                {'name': 'Pneumococcal', 'doses': 1, 'last_dose': self.fake.date_between(start_date='-5y', end_date='today')}
            ]
            immunizations.extend(adult_vaccines)
        return immunizations

    def _generate_social_determinants(self) -> Dict[str, Any]:
        """Generate social determinants of health information."""
        return {
            'food_security': random.choice(['Secure', 'Marginally Secure', 'Insecure']),
            'housing_stability': random.choice(['Stable', 'Marginally Stable', 'Unstable']),
            'transportation_access': random.choice(['Good', 'Limited', 'Poor']),
            'social_support': random.choice(['Strong', 'Moderate', 'Limited']),
            'community_involvement': random.choice(['Active', 'Moderate', 'Limited']),
            'health_literacy': random.choice(['High', 'Moderate', 'Low']),
            'access_to_care': random.choice(['Good', 'Limited', 'Poor']),
            'environmental_factors': random.sample(['Air quality', 'Water quality', 'Noise pollution', 'Safety'], 
                                                random.randint(0, 3))
        }

    def _generate_preventive_care(self, age: int) -> Dict[str, Any]:
        """Generate preventive care information based on age."""
        preventive_care = {
            'last_physical': self.fake.date_between(start_date='-2y', end_date='today'),
            'last_dental': self.fake.date_between(start_date='-2y', end_date='today'),
            'last_vision': self.fake.date_between(start_date='-2y', end_date='today'),
            'screening_tests': []
        }
        
        if age >= 50:
            preventive_care['screening_tests'].extend([
                {'name': 'Colonoscopy', 'last_test': self.fake.date_between(start_date='-5y', end_date='today')},
                {'name': 'Mammogram', 'last_test': self.fake.date_between(start_date='-2y', end_date='today')},
                {'name': 'Prostate', 'last_test': self.fake.date_between(start_date='-2y', end_date='today')}
            ])
        
        return preventive_care

    def _generate_risk_factors(self) -> List[Dict[str, Any]]:
        """Generate health risk factors."""
        risk_factors = []
        possible_factors = [
            {'factor': 'Smoking', 'level': random.choice(['None', 'Former', 'Current'])},
            {'factor': 'Alcohol', 'level': random.choice(['None', 'Moderate', 'Heavy'])},
            {'factor': 'Physical Activity', 'level': random.choice(['Sedentary', 'Moderate', 'Active'])},
            {'factor': 'Diet', 'level': random.choice(['Poor', 'Fair', 'Good'])},
            {'factor': 'Stress', 'level': random.choice(['Low', 'Moderate', 'High'])},
            {'factor': 'Sleep', 'level': random.choice(['Poor', 'Fair', 'Good'])}
        ]
        
        for factor in possible_factors:
            if random.random() < 0.7:  # 70% chance to include each factor
                risk_factors.append(factor)
        
        return risk_factors

    def _generate_cultural_preferences(self) -> Dict[str, Any]:
        """Generate cultural preferences information."""
        return {
            'religious_preferences': random.choice(['None', 'Christian', 'Muslim', 'Hindu', 'Buddhist', 'Jewish', 'Other']),
            'dietary_restrictions': random.sample(['None', 'Vegetarian', 'Vegan', 'Halal', 'Kosher', 'Gluten-free'], 
                                               random.randint(1, 3)),
            'cultural_practices': random.sample(['Traditional Medicine', 'Spiritual Healing', 'Meditation', 'Yoga'], 
                                             random.randint(0, 2)),
            'language_preference': random.choice(['English', 'Spanish', 'Hindi', 'Mandarin', 'French', 'Arabic']),
            'family_decision_making': random.choice(['Individual', 'Family', 'Community'])
        }

    def _generate_accessibility_needs(self) -> List[Dict[str, Any]]:
        """Generate accessibility needs information."""
        needs = []
        possible_needs = [
            {'type': 'Mobility', 'requirement': 'Wheelchair Access'},
            {'type': 'Vision', 'requirement': 'Large Print Materials'},
            {'type': 'Hearing', 'requirement': 'Sign Language Interpreter'},
            {'type': 'Communication', 'requirement': 'Speech-to-Text'},
            {'type': 'Cognitive', 'requirement': 'Simplified Instructions'}
        ]
        
        for need in possible_needs:
            if random.random() < 0.2:  # 20% chance for each need
                needs.append(need)
        
        return needs

    def _generate_allergies(self) -> List[Dict[str, str]]:
        """Generate realistic allergies."""
        common_allergies = [
            {'type': 'Medication', 'name': 'Penicillin', 'severity': 'Severe'},
            {'type': 'Food', 'name': 'Peanuts', 'severity': 'Moderate'},
            {'type': 'Environmental', 'name': 'Pollen', 'severity': 'Mild'},
            {'type': 'Medication', 'name': 'Sulfa drugs', 'severity': 'Moderate'},
            {'type': 'Food', 'name': 'Shellfish', 'severity': 'Severe'}
        ]
        return random.sample(common_allergies, random.randint(0, 2))

    def _generate_family_history(self) -> List[Dict[str, str]]:
        """Generate realistic family medical history."""
        conditions = [
            {'condition': 'Heart Disease', 'relation': 'Parent'},
            {'condition': 'Diabetes', 'relation': 'Sibling'},
            {'condition': 'Cancer', 'relation': 'Grandparent'},
            {'condition': 'Hypertension', 'relation': 'Parent'},
            {'condition': 'Asthma', 'relation': 'Sibling'}
        ]
        return random.sample(conditions, random.randint(0, 3))

    def generate_medical_history(self, patient_df):
        """Generate comprehensive medical history for patients."""
        medical_histories = []
        
        for _, patient in patient_df.iterrows():
            try:
                # Get age-appropriate conditions
                conditions = self._get_age_appropriate_conditions(patient['age'])
                if not conditions:
                    print(f"Warning: No conditions found for age {patient['age']}, using default conditions")
                    conditions = ['Hypertension', 'Type 2 Diabetes', 'Asthma', 'Anxiety', 'Depression']
                
                # Select primary and secondary conditions
                primary_condition = random.choice(conditions)
                remaining_conditions = [c for c in conditions if c != primary_condition]
                secondary_conditions = random.sample(remaining_conditions, min(2, len(remaining_conditions)))
                
                # Generate diagnosis dates
                diagnosis_date = datetime.now() - timedelta(days=random.randint(30, 3650))  # 1-10 years ago
                
                # Get condition information
                condition_info = self.medical_knowledge['conditions'][primary_condition]
                
                # Generate medical history
                medical_history = {
                    'patient_id': patient['patient_id'],
                    'primary_condition': primary_condition,
                    'secondary_conditions': secondary_conditions,
                    'diagnosis_date': diagnosis_date,
                    'severity': random.choice(condition_info['severity_levels']),
                    'symptoms': random.sample(condition_info['symptoms'], min(3, len(condition_info['symptoms']))),
                    'risk_factors': random.sample(condition_info['risk_factors'], min(3, len(condition_info['risk_factors']))),
                    'treatments': random.sample(condition_info['treatments'], min(3, len(condition_info['treatments']))),
                    'medications': random.sample(condition_info['medications'], min(2, len(condition_info['medications']))),
                    'icd10_code': condition_info['icd10_code'],
                    'follow_up_frequency': condition_info['follow_up_frequency'],
                    'complications': random.sample(condition_info['complications'], min(2, len(condition_info['complications']))),
                    'preventive_measures': random.sample(condition_info['preventive_measures'], min(3, len(condition_info['preventive_measures']))),
                    'treatment_history': self._generate_treatment_history(condition_info, diagnosis_date),
                    'lab_results': self._generate_lab_results(primary_condition),
                    'imaging_studies': self._generate_imaging_studies(primary_condition),
                    'specialist_consultations': self._generate_specialist_consultations(primary_condition),
                    'hospitalizations': self._generate_hospitalizations(primary_condition, diagnosis_date),
                    'emergency_visits': self._generate_emergency_visits(primary_condition, diagnosis_date),
                    'complications': self._generate_complications(primary_condition),
                    'follow_up_care': self._generate_follow_up_care(primary_condition, diagnosis_date)
                }
                
                medical_histories.append(medical_history)
                
            except Exception as e:
                print(f"Error generating medical history for patient {patient['patient_id']}: {str(e)}")
                continue
        
        return pd.DataFrame(medical_histories)

    def _generate_treatment_history(self, condition_info: Dict[str, Any], diagnosis_date: datetime) -> List[Dict[str, Any]]:
        """Generate detailed treatment history."""
        treatments = []
        num_treatments = random.randint(1, 3)
        
        for _ in range(num_treatments):
            treatment = {
                'type': random.choice(condition_info['treatments']),
                'start_date': self.fake.date_between(start_date=diagnosis_date, end_date='today'),
                'end_date': None if random.random() < 0.7 else self.fake.date_between(start_date='today', end_date='+1y'),
                'provider': self.fake.name(),
                'location': f"Room {random.randint(100, 999)}",
                'frequency': random.choice(['Daily', 'Weekly', 'Monthly', 'As needed']),
                'response': random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
                'side_effects': random.sample(['None', 'Mild', 'Moderate', 'Severe'], random.randint(1, 2)),
                'compliance': random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
                'notes': self._generate_treatment_notes()
            }
            treatments.append(treatment)
        
        return treatments

    def _generate_lab_results(self, condition: str) -> List[Dict[str, Any]]:
        """Generate laboratory test results."""
        lab_tests = []
        num_tests = random.randint(1, 4)
        
        for _ in range(num_tests):
            test = {
                'test_name': random.choice(['Blood Test', 'Urine Test', 'Culture', 'Biopsy']),
                'date': self.fake.date_between(start_date='-1y', end_date='today'),
                'results': {
                    'value': round(random.uniform(0, 100), 2),
                    'unit': random.choice(['mg/dL', 'mmol/L', 'g/L', 'U/L']),
                    'reference_range': f"{round(random.uniform(0, 50), 2)}-{round(random.uniform(50, 100), 2)}",
                    'status': random.choice(['Normal', 'Abnormal', 'Critical'])
                },
                'performing_lab': self.fake.company(),
                'ordering_physician': self.fake.name(),
                'notes': self._generate_lab_notes()
            }
            lab_tests.append(test)
        
        return lab_tests

    def _generate_imaging_studies(self, condition: str) -> List[Dict[str, Any]]:
        """Generate imaging study results."""
        imaging_studies = []
        num_studies = random.randint(1, 3)
        
        for _ in range(num_studies):
            study = {
                'type': random.choice(['X-Ray', 'CT Scan', 'MRI', 'Ultrasound', 'PET Scan']),
                'date': self.fake.date_between(start_date='-1y', end_date='today'),
                'findings': self._generate_imaging_findings(condition),
                'impression': self._generate_imaging_impression(condition),
                'performing_facility': self.fake.company(),
                'radiologist': self.fake.name(),
                'ordering_physician': self.fake.name(),
                'follow_up_recommended': random.random() < 0.3
            }
            imaging_studies.append(study)
        
        return imaging_studies

    def _generate_specialist_consultations(self, condition: str) -> List[Dict[str, Any]]:
        """Generate specialist consultation records."""
        consultations = []
        num_consultations = random.randint(1, 3)
        
        for _ in range(num_consultations):
            consultation = {
                'specialist': self.fake.name(),
                'specialty': random.choice(['Cardiology', 'Neurology', 'Orthopedics', 'Endocrinology']),
                'date': self.fake.date_between(start_date='-1y', end_date='today'),
                'reason': self._generate_consultation_reason(condition),
                'assessment': self._generate_consultation_assessment(condition),
                'recommendations': self._generate_consultation_recommendations(),
                'follow_up_required': random.random() < 0.7,
                'follow_up_date': self.fake.date_between(start_date='today', end_date='+3m')
            }
            consultations.append(consultation)
        
        return consultations

    def _generate_hospitalizations(self, condition: str, diagnosis_date: datetime) -> List[Dict[str, Any]]:
        """Generate hospitalization records."""
        hospitalizations = []
        num_hospitalizations = random.randint(0, 2)
        
        for _ in range(num_hospitalizations):
            admission_date = self.fake.date_between(start_date=diagnosis_date, end_date='today')
            discharge_date = self.fake.date_between(start_date=admission_date, end_date='+14d')
            
            hospitalization = {
                'admission_date': admission_date,
                'discharge_date': discharge_date,
                'length_of_stay': (discharge_date - admission_date).days,
                'admission_reason': self._generate_admission_reason(condition),
                'discharge_diagnosis': self._generate_discharge_diagnosis(condition),
                'treatments_received': self._generate_hospital_treatments(),
                'complications': self._generate_hospital_complications(),
                'discharge_disposition': random.choice(['Home', 'Rehabilitation', 'Skilled Nursing', 'Hospice']),
                'follow_up_required': random.random() < 0.8,
                'readmission_within_30_days': random.random() < 0.1
            }
            hospitalizations.append(hospitalization)
        
        return hospitalizations

    def _generate_emergency_visits(self, condition: str, diagnosis_date: datetime) -> List[Dict[str, Any]]:
        """Generate emergency visit records."""
        emergency_visits = []
        num_visits = random.randint(0, 2)
        
        for _ in range(num_visits):
            visit = {
                'date': self.fake.date_between(start_date=diagnosis_date, end_date='today'),
                'chief_complaint': self._generate_chief_complaint(condition),
                'triage_level': random.choice(['1-Immediate', '2-Emergent', '3-Urgent', '4-Less Urgent', '5-Non-urgent']),
                'vital_signs': self._generate_emergency_vitals(),
                'treatments': self._generate_emergency_treatments(),
                'disposition': random.choice(['Discharged', 'Admitted', 'Transferred', 'Left AMA']),
                'follow_up_required': random.random() < 0.6,
                'follow_up_date': self.fake.date_between(start_date='today', end_date='+7d')
            }
            emergency_visits.append(visit)
        
        return emergency_visits

    def _generate_complications(self, condition: str) -> List[Dict[str, Any]]:
        """Generate complication records."""
        complications = []
        num_complications = random.randint(0, 2)
        
        for _ in range(num_complications):
            complication = {
                'type': random.choice(['Acute', 'Chronic', 'Recurrent']),
                'description': self._generate_complication_description(condition),
                'onset_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'severity': random.choice(['Mild', 'Moderate', 'Severe']),
                'treatment': self._generate_complication_treatment(),
                'resolution': random.choice(['Resolved', 'Ongoing', 'Recurrent']),
                'impact_on_condition': random.choice(['Worsened', 'No Change', 'Improved'])
            }
            complications.append(complication)
        
        return complications

    def _generate_follow_up_care(self, condition: str, diagnosis_date: datetime) -> Dict[str, Any]:
        """Generate follow-up care plan."""
        return {
            'frequency': random.choice(['Weekly', 'Monthly', 'Every 3 months', 'Every 6 months', 'Annually']),
            'next_appointment': self.fake.date_between(start_date='today', end_date='+6m'),
            'monitoring_parameters': self._generate_monitoring_parameters(condition),
            'lifestyle_modifications': self._generate_lifestyle_modifications(),
            'medication_adherence_plan': self._generate_adherence_plan(),
            'support_services': self._generate_support_services(),
            'emergency_instructions': self._generate_emergency_instructions(),
            'goals': self._generate_treatment_goals()
        }

    def _generate_treatment_notes(self) -> str:
        """Generate treatment notes."""
        templates = [
            "Patient responding well to treatment",
            "Treatment plan adjusted based on response",
            "Side effects managed with supportive care",
            "Treatment compliance needs improvement",
            "Treatment goals being met"
        ]
        return random.choice(templates)

    def _generate_lab_notes(self) -> str:
        """Generate laboratory test notes."""
        templates = [
            "Results within normal range",
            "Results indicate improvement",
            "Results require follow-up",
            "Results suggest treatment adjustment needed",
            "Results stable compared to previous"
        ]
        return random.choice(templates)

    def _generate_imaging_findings(self, condition: str) -> str:
        """Generate imaging findings."""
        templates = {
            'Heart Disease': [
                "Normal cardiac silhouette",
                "Cardiomegaly noted",
                "Pulmonary congestion present",
                "Normal pulmonary vascularity"
            ],
            'Cancer': [
                "No evidence of metastasis",
                "Lesion size stable",
                "New nodule identified",
                "Previous findings resolved"
            ]
        }
        return random.choice(templates.get(condition, ["Normal study", "No significant findings", "Stable compared to previous"]))

    def _generate_imaging_impression(self, condition: str) -> str:
        """Generate imaging impression."""
        templates = {
            'Heart Disease': [
                "Normal cardiac study",
                "Mild cardiomegaly",
                "Signs of heart failure",
                "Stable cardiac findings"
            ],
            'Cancer': [
                "No evidence of disease progression",
                "Stable disease",
                "New finding requiring follow-up",
                "Complete response to treatment"
            ]
        }
        return random.choice(templates.get(condition, ["Normal study", "No significant findings", "Stable compared to previous"]))

    def _generate_consultation_reason(self, condition: str) -> str:
        """Generate consultation reason."""
        templates = {
            'Heart Disease': [
                "Evaluation of chest pain",
                "Management of heart failure",
                "Assessment of arrhythmia",
                "Pre-operative cardiac clearance"
            ],
            'Cancer': [
                "Treatment planning",
                "Second opinion",
                "Management of complications",
                "Follow-up care"
            ]
        }
        return random.choice(templates.get(condition, ["Specialist evaluation", "Treatment planning", "Management of condition"]))

    def _generate_consultation_assessment(self, condition: str) -> str:
        """Generate consultation assessment."""
        templates = {
            'Heart Disease': [
                "Stable cardiac condition",
                "Worsening heart failure",
                "New arrhythmia identified",
                "Improved cardiac function"
            ],
            'Cancer': [
                "Stable disease",
                "Progressive disease",
                "Treatment response noted",
                "New concerns identified"
            ]
        }
        return random.choice(templates.get(condition, ["Condition stable", "Improvement noted", "New concerns identified"]))

    def _generate_consultation_recommendations(self) -> List[str]:
        """Generate consultation recommendations."""
        recommendations = []
        possible_recommendations = [
            "Continue current treatment",
            "Adjust medication dosage",
            "Add new medication",
            "Schedule follow-up imaging",
            "Refer to additional specialist",
            "Implement lifestyle modifications"
        ]
        num_recommendations = random.randint(1, 3)
        return random.sample(possible_recommendations, num_recommendations)

    def _generate_admission_reason(self, condition: str) -> str:
        """Generate admission reason."""
        templates = {
            'Heart Disease': [
                "Acute chest pain",
                "Worsening heart failure",
                "Unstable arrhythmia",
                "Post-cardiac procedure"
            ],
            'Cancer': [
                "Treatment administration",
                "Management of complications",
                "Pain control",
                "Post-surgical care"
            ]
        }
        return random.choice(templates.get(condition, ["Acute condition", "Treatment administration", "Complication management"]))

    def _generate_discharge_diagnosis(self, condition: str) -> str:
        """Generate discharge diagnosis."""
        templates = {
            'Heart Disease': [
                "Stable heart failure",
                "Resolved chest pain",
                "Controlled arrhythmia",
                "Post-procedure recovery"
            ],
            'Cancer': [
                "Stable disease",
                "Treatment completed",
                "Complications resolved",
                "Post-surgical recovery"
            ]
        }
        return random.choice(templates.get(condition, ["Condition stable", "Treatment completed", "Recovery in progress"]))

    def _generate_hospital_treatments(self) -> List[Dict[str, Any]]:
        """Generate hospital treatments."""
        treatments = []
        num_treatments = random.randint(1, 4)
        
        for _ in range(num_treatments):
            treatment = {
                'type': random.choice(['Medication', 'Procedure', 'Therapy', 'Surgery']),
                'date': self.fake.date_between(start_date='-14d', end_date='today'),
                'provider': self.fake.name(),
                'response': random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
                'complications': random.random() < 0.2
            }
            treatments.append(treatment)
        
        return treatments

    def _generate_hospital_complications(self) -> List[Dict[str, Any]]:
        """Generate hospital complications."""
        complications = []
        if random.random() < 0.3:  # 30% chance of complications
            num_complications = random.randint(1, 2)
            for _ in range(num_complications):
                complication = {
                    'type': random.choice(['Infection', 'Bleeding', 'Reaction', 'Other']),
                    'severity': random.choice(['Mild', 'Moderate', 'Severe']),
                    'resolved': random.random() < 0.8
                }
                complications.append(complication)
        return complications

    def _generate_chief_complaint(self, condition: str) -> str:
        """Generate chief complaint."""
        templates = {
            'Heart Disease': [
                "Chest pain",
                "Shortness of breath",
                "Palpitations",
                "Swelling"
            ],
            'Cancer': [
                "Pain",
                "Fever",
                "Weakness",
                "Nausea"
            ]
        }
        return random.choice(templates.get(condition, ["Pain", "Fever", "Weakness", "Other symptoms"]))

    def _generate_emergency_vitals(self) -> Dict[str, Any]:
        """Generate emergency vital signs."""
        return {
            'blood_pressure': f"{random.randint(90, 160)}/{random.randint(60, 100)}",
            'heart_rate': random.randint(60, 120),
            'respiratory_rate': random.randint(12, 24),
            'temperature': round(random.uniform(97.0, 101.0), 1),
            'oxygen_saturation': random.randint(90, 100)
        }

    def _generate_emergency_treatments(self) -> List[Dict[str, Any]]:
        """Generate emergency treatments."""
        treatments = []
        num_treatments = random.randint(1, 3)
        
        for _ in range(num_treatments):
            treatment = {
                'type': random.choice(['Medication', 'IV Fluids', 'Oxygen', 'Procedure']),
                'time': self.fake.time(),
                'provider': self.fake.name(),
                'response': random.choice(['Good', 'Fair', 'Poor'])
            }
            treatments.append(treatment)
        
        return treatments

    def _generate_complication_description(self, condition: str) -> str:
        """Generate complication description."""
        templates = {
            'Heart Disease': [
                "Worsening heart failure",
                "New arrhythmia",
                "Medication side effect",
                "Infection"
            ],
            'Cancer': [
                "Treatment side effect",
                "Infection",
                "Pain exacerbation",
                "Metabolic complication"
            ]
        }
        return random.choice(templates.get(condition, ["Treatment complication", "Disease progression", "New symptom"]))

    def _generate_complication_treatment(self) -> Dict[str, Any]:
        """Generate complication treatment."""
        return {
            'type': random.choice(['Medication', 'Procedure', 'Therapy']),
            'start_date': self.fake.date_between(start_date='-1y', end_date='today'),
            'provider': self.fake.name(),
            'response': random.choice(['Good', 'Fair', 'Poor'])
        }

    def _generate_monitoring_parameters(self, condition: str) -> List[Dict[str, Any]]:
        """Generate monitoring parameters."""
        parameters = []
        if condition == 'Heart Disease':
            parameters.extend([
                {'parameter': 'Blood Pressure', 'frequency': 'Daily'},
                {'parameter': 'Heart Rate', 'frequency': 'Daily'},
                {'parameter': 'Weight', 'frequency': 'Daily'},
                {'parameter': 'Symptoms', 'frequency': 'As needed'}
            ])
        elif condition == 'Diabetes':
            parameters.extend([
                {'parameter': 'Blood Glucose', 'frequency': 'Multiple times daily'},
                {'parameter': 'HbA1c', 'frequency': 'Every 3 months'},
                {'parameter': 'Foot Exam', 'frequency': 'Daily'},
                {'parameter': 'Weight', 'frequency': 'Weekly'}
            ])
        else:
            parameters.extend([
                {'parameter': 'Symptoms', 'frequency': 'Daily'},
                {'parameter': 'Medication Side Effects', 'frequency': 'As needed'},
                {'parameter': 'Quality of Life', 'frequency': 'Weekly'}
            ])
        return parameters

    def _generate_lifestyle_modifications(self) -> List[Dict[str, Any]]:
        """Generate lifestyle modifications."""
        modifications = []
        possible_modifications = [
            {'type': 'Diet', 'recommendation': 'Follow prescribed diet plan'},
            {'type': 'Exercise', 'recommendation': 'Regular physical activity'},
            {'type': 'Stress Management', 'recommendation': 'Practice relaxation techniques'},
            {'type': 'Sleep', 'recommendation': 'Maintain regular sleep schedule'},
            {'type': 'Smoking', 'recommendation': 'Smoking cessation'},
            {'type': 'Alcohol', 'recommendation': 'Limit alcohol consumption'}
        ]
        num_modifications = random.randint(2, 4)
        return random.sample(possible_modifications, num_modifications)

    def _generate_adherence_plan(self) -> Dict[str, Any]:
        """Generate medication adherence plan."""
        return {
            'reminder_system': random.choice(['Phone App', 'Pill Box', 'Calendar', 'Family Support']),
            'refill_reminder': random.choice(['Automatic', 'Manual', 'Pharmacy']),
            'barriers_identified': random.sample(['Cost', 'Side Effects', 'Complexity', 'Forgetting'], 
                                              random.randint(0, 2)),
            'strategies': random.sample(['Simplified Schedule', 'Family Support', 'Pharmacy Support', 'Education'], 
                                     random.randint(1, 3))
        }

    def _generate_support_services(self) -> List[Dict[str, Any]]:
        """Generate support services."""
        services = []
        possible_services = [
            {'type': 'Social Work', 'purpose': 'Resource coordination'},
            {'type': 'Nutrition', 'purpose': 'Dietary planning'},
            {'type': 'Physical Therapy', 'purpose': 'Mobility improvement'},
            {'type': 'Mental Health', 'purpose': 'Emotional support'},
            {'type': 'Support Group', 'purpose': 'Peer support'}
        ]
        num_services = random.randint(1, 3)
        return random.sample(possible_services, num_services)

    def _generate_emergency_instructions(self) -> Dict[str, Any]:
        """Generate emergency instructions."""
        return {
            'warning_signs': random.sample(['Severe pain', 'Difficulty breathing', 'High fever', 'Uncontrolled bleeding'], 
                                         random.randint(2, 4)),
            'emergency_contact': self.fake.phone_number(),
            'emergency_location': 'Nearest Emergency Department',
            'action_plan': random.choice(['Call 911', 'Go to ER', 'Call Provider'])
        }

    def _generate_treatment_goals(self) -> List[Dict[str, Any]]:
        """Generate treatment goals."""
        goals = []
        possible_goals = [
            {'goal': 'Symptom Control', 'timeline': 'Ongoing'},
            {'goal': 'Functional Improvement', 'timeline': '3 months'},
            {'goal': 'Quality of Life', 'timeline': 'Ongoing'},
            {'goal': 'Disease Management', 'timeline': 'Long-term'},
            {'goal': 'Prevention', 'timeline': 'Ongoing'}
        ]
        num_goals = random.randint(2, 4)
        return random.sample(possible_goals, num_goals)

    def generate_appointments(self, patient_df, num_appointments=2000):
        """Generate comprehensive appointment data with enhanced hospital scheduling scenarios."""
        appointments = []
        
        departments = list(self.hospital_infrastructure['departments'].keys())
        
        for _ in range(num_appointments):
            patient = patient_df.sample(n=1).iloc[0]
            department = random.choice(departments)
            doctors = self.hospital_infrastructure['departments'][department]['doctors']
            appointment_type = self._generate_appointment_type(department)
            appointment_time = self._generate_appointment_time()
            duration = self._get_appointment_duration(appointment_type)
            status = self._generate_appointment_status(appointment_time)
            
            appointment = {
                'appointment_id': f"APT{random.randint(100000, 999999)}",
                'patient_id': patient['patient_id'],
                'department': department,
                'appointment_type': appointment_type,
                'scheduled_time': appointment_time,
                'duration_minutes': duration,
                'status': status,
                'reason': self._generate_appointment_reason(department),
                'provider': {
                    'name': random.choice(doctors),
                    'title': random.choice(['MD', 'DO', 'NP', 'PA']),
                    'specialty': random.choice(self.hospital_infrastructure['departments'][department]['specialties']),
                    'department': department
                },
                'location': {
                    'building': random.choice(['Main', 'East', 'West', 'North', 'South']),
                    'floor': random.randint(1, 5),
                    'room': f"{random.randint(100, 999)}",
                    'wing': random.choice(['A', 'B', 'C', 'D'])
                },
                'check_in': {
                    'time': appointment_time - timedelta(minutes=15) if status != 'Cancelled' else None,
                    'status': random.choice(['On Time', 'Late', 'Early']) if status == 'Completed' else None,
                    'wait_time': random.randint(0, 45) if status == 'Completed' else None
                },
                'vitals': {
                    'blood_pressure': f"{random.randint(90, 140)}/{random.randint(60, 90)}",
                    'heart_rate': random.randint(60, 100),
                    'temperature': round(random.uniform(97.0, 99.0), 1),
                    'oxygen_saturation': random.randint(95, 100),
                    'weight': round(random.uniform(100, 250), 1),
                    'height': round(random.uniform(60, 76), 1)
                } if status == 'Completed' else None,
                'notes': self._generate_appointment_notes(appointment_type),
                'preparation_instructions': self._generate_preparation_instructions(appointment_type),
                'follow_up': {
                    'needed': random.random() < 0.7,
                    'recommended_date': appointment_time + timedelta(days=random.randint(7, 90)) if random.random() < 0.7 else None,
                    'reason': random.choice(['Routine Follow-up', 'Test Results Review', 'Treatment Adjustment', 'Progress Check'])
                },
                'insurance': {
                    'verified': random.random() < 0.9,
                    'coverage_status': random.choice(['Covered', 'Partial', 'Not Covered']),
                    'prior_authorization': random.choice(['Required', 'Not Required', 'Pending', 'Approved', 'Denied']),
                    'copay': random.randint(0, 100) if random.random() < 0.8 else None
                },
                'billing': {
                    'estimated_cost': random.randint(100, 1000),
                    'insurance_estimated_payment': random.randint(50, 800),
                    'patient_estimated_responsibility': random.randint(0, 200),
                    'billing_code': f"CPT{random.randint(10000, 99999)}"
                },
                'quality_metrics': {
                    'patient_satisfaction': random.randint(1, 5) if status == 'Completed' else None,
                    'provider_satisfaction': random.randint(1, 5) if status == 'Completed' else None,
                    'wait_time_satisfaction': random.randint(1, 5) if status == 'Completed' else None,
                    'communication_quality': random.randint(1, 5) if status == 'Completed' else None
                },
                'resources': {
                    'equipment_needed': random.sample(['Stethoscope', 'Blood Pressure Cuff', 'EKG Machine', 'Ultrasound', 'X-Ray'], 
                                                    k=random.randint(0, 3)),
                    'staff_needed': random.sample(['Nurse', 'Medical Assistant', 'Technician', 'Interpreter'], 
                                                k=random.randint(1, 3)),
                    'room_type': random.choice(['Standard', 'Procedure', 'Consultation', 'Examination'])
                },
                'documentation': {
                    'chief_complaint': self._generate_chief_complaint(random.choice(['Hypertension', 'Diabetes', 'Asthma', 'Arthritis'])),
                    'assessment': random.choice(['Stable', 'Improving', 'Worsening', 'New Onset']),
                    'plan': random.choice(['Continue Current Treatment', 'Adjust Medication', 'Order Tests', 'Refer to Specialist']),
                    'orders': random.sample(['Lab Work', 'Imaging', 'Medication', 'Therapy', 'Follow-up'], 
                                          k=random.randint(0, 3))
                },
                'scheduling': {
                    'scheduled_by': random.choice(['Patient', 'Provider', 'Nurse', 'Scheduler']),
                    'scheduling_method': random.choice(['Phone', 'Online', 'In Person', 'Mobile App']),
                    'reminder_sent': random.random() < 0.9,
                    'confirmation_received': random.random() < 0.8,
                    'rescheduled_count': random.randint(0, 3)
                },
                'cancellation': {
                    'reason': random.choice(['Patient Request', 'Provider Request', 'Weather', 'Emergency', 'No Show']) 
                             if status == 'Cancelled' else None,
                    'notice_given': random.choice(['24+ hours', '12-24 hours', '<12 hours', 'No notice']) 
                                  if status == 'Cancelled' else None
                }
            }
            
            appointments.append(appointment)
            
        return pd.DataFrame(appointments)

    def _generate_appointment_time(self) -> datetime:
        """Generate realistic appointment time."""
        # Generate time between 8 AM and 5 PM on weekdays
        date = self.fake.date_between(start_date='-30d', end_date='+30d')
        while date.weekday() >= 5:  # Skip weekends
            date = self.fake.date_between(start_date='-30d', end_date='+30d')
        
        hour = random.randint(8, 16)
        minute = random.choice([0, 15, 30, 45])
        return datetime.combine(date, datetime.min.time().replace(hour=hour, minute=minute))

    def _generate_appointment_type(self, department: str) -> str:
        """Generate appropriate appointment type for department."""
        types = {
            'Cardiology': ['Consultation', 'EKG', 'Stress Test', 'Echocardiogram'],
            'Neurology': ['Consultation', 'MRI', 'EEG', 'Nerve Study'],
            'Orthopedics': ['Consultation', 'X-ray', 'Physical Therapy', 'Joint Injection']
        }
        return random.choice(types.get(department, ['Consultation', 'Follow-up', 'Check-up']))

    def _get_appointment_duration(self, appointment_type: str) -> int:
        """Get appropriate duration for appointment type."""
        durations = {
            'Consultation': 30,
            'Follow-up': 15,
            'Check-up': 45,
            'EKG': 20,
            'Stress Test': 60,
            'MRI': 45,
            'X-ray': 15,
            'Physical Therapy': 60
        }
        return durations.get(appointment_type, 30)

    def _generate_appointment_status(self, appointment_date: datetime) -> str:
        """Generate realistic appointment status."""
        if appointment_date < datetime.now():
            return random.choice(['Completed', 'No-Show', 'Cancelled'])
        return 'Scheduled'

    def _generate_appointment_reason(self, department: str) -> str:
        """Generate department-specific appointment reasons."""
        reasons = {
            'Cardiology': [
                'Chest pain evaluation',
                'Heart rhythm check',
                'Blood pressure monitoring',
                'Cardiac stress test'
            ],
            'Neurology': [
                'Headache evaluation',
                'Memory assessment',
                'Nerve pain consultation',
                'Movement disorder check'
            ],
            'Orthopedics': [
                'Joint pain evaluation',
                'Sports injury follow-up',
                'Fracture check',
                'Physical therapy assessment'
            ]
        }
        return random.choice(reasons.get(department, [
            'Regular Check-up',
            'Follow-up Visit',
            'New Symptoms',
            'Test Results Review'
        ]))

    def _generate_appointment_notes(self, appointment_type: str) -> str:
        """Generate relevant appointment notes."""
        templates = {
            'Consultation': [
                "Initial consultation for new patient",
                "Follow-up consultation for ongoing treatment",
                "Second opinion consultation"
            ],
            'Follow-up': [
                "Progress check after treatment",
                "Medication adjustment follow-up",
                "Post-procedure follow-up"
            ],
            'Check-up': [
                "Annual health check-up",
                "Routine wellness examination",
                "Preventive health screening"
            ]
        }
        return random.choice(templates.get(appointment_type, ["Regular appointment"]))

    def _generate_preparation_instructions(self, appointment_type: str) -> str:
        """Generate preparation instructions for appointment."""
        instructions = {
            'MRI': "Please arrive 30 minutes early. No food 4 hours before appointment.",
            'Stress Test': "Wear comfortable clothing and shoes. No caffeine 24 hours before test.",
            'Blood Test': "Fasting required for 12 hours before appointment.",
            'Consultation': "Bring all current medications and medical records.",
            'Physical Therapy': "Wear loose, comfortable clothing suitable for exercise."
        }
        return instructions.get(appointment_type, "No special preparation required.")

    def generate_staff_schedule(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate comprehensive staff schedule with enhanced hospital staffing scenarios."""
        schedules = []
        
        # Gather all staff members
        staff_members = []
        
        # Add doctors from each department
        for dept, info in self.hospital_infrastructure['departments'].items():
            for doctor in info['doctors']:
                staff_members.append({
                    'staff_id': f"DOC{len(staff_members)+1:04d}",
                    'name': doctor,
                    'role': 'Doctor',
                    'specialty': dept,
                    'title': random.choice(['Attending', 'Resident', 'Fellow']),
                    'department': dept,
                    'credentials': random.choice(['MD', 'DO'])
                })
        
        # Add nurses
        for dept, info in self.hospital_infrastructure['departments'].items():
            for nurse in info['nurses']:
                staff_members.append({
                    'staff_id': f"NUR{len(staff_members)+1:04d}",
                    'name': nurse,
                    'role': 'Nurse',
                    'specialty': dept,
                    'title': random.choice(['RN', 'LPN', 'NP']),
                    'department': dept,
                    'credentials': random.choice(['RN', 'LPN', 'NP'])
                })
        
        # Add support staff
        support_roles = ['Technician', 'Administrator', 'Coordinator', 'Assistant']
        for role in support_roles:
            for _ in range(5):  # 5 staff members per role
                staff_members.append({
                    'staff_id': f"SUP{len(staff_members)+1:04d}",
                    'name': self.fake.name(),
                    'role': role,
                    'specialty': random.choice(['General', 'Specialized']),
                    'title': role,
                    'department': random.choice(list(self.hospital_infrastructure['departments'].keys())),
                    'credentials': random.choice(['Certified', 'Licensed', 'Trained'])
                })
        
        # Generate schedules for each staff member
        for staff in staff_members:
            current_date = start_date
            while current_date <= end_date:
                # Skip weekends for non-essential staff
                if current_date.weekday() >= 5 and staff['role'] not in ['Doctor', 'Nurse']:
                    current_date += timedelta(days=1)
                    continue
                
                # Define shift types and hours based on role
                if staff['role'] == 'Doctor':
                    shift_types = ['Day', 'Night', 'On-Call']
                    shift_hours = {
                        'Day': (8, 20),    # 8 AM to 8 PM
                        'Night': (20, 8),  # 8 PM to 8 AM
                        'On-Call': (0, 23) # 24-hour on-call (using 23:59 as end time)
                    }
                elif staff['role'] == 'Nurse':
                    shift_types = ['Day', 'Evening', 'Night']
                    shift_hours = {
                        'Day': (7, 19),     # 7 AM to 7 PM
                        'Evening': (15, 3), # 3 PM to 3 AM
                        'Night': (19, 7)    # 7 PM to 7 AM
                    }
                else:
                    shift_types = ['Day']
                    shift_hours = {
                        'Day': (9, 17)  # 9 AM to 5 PM
                    }
                
                # Generate schedule for the day
                shift_type = random.choice(shift_types)
                hours = shift_hours[shift_type]
                
                # Handle overnight shifts
                if hours[1] < hours[0]:  # Overnight shift
                    end_time = current_date.replace(hour=hours[1], minute=0) + timedelta(days=1)
                else:
                    end_time = current_date.replace(hour=hours[1], minute=0)
                
                schedule = {
                    'schedule_id': f"SCH{len(schedules)+1:06d}",
                    'staff_id': staff['staff_id'],
                    'date': current_date.date(),
                    'shift_type': shift_type,
                    'start_time': current_date.replace(hour=hours[0], minute=0),
                    'end_time': end_time,
                    'location': staff['department'],
                    'assignments': {
                        'primary_role': staff['role'],
                        'secondary_roles': random.sample(['Patient Care', 'Documentation', 'Training', 'Quality Assurance'], 
                                                      k=random.randint(1, 3)),
                        'patient_load': random.randint(1, 10) if staff['role'] in ['Doctor', 'Nurse'] else 0,
                        'supervision': random.choice(['None', 'Supervising', 'Supervised']),
                        'special_assignments': random.sample(['Emergency Response', 'Code Team', 'Rapid Response'], 
                                                          k=random.randint(0, 2))
                    },
                    'breaks': {
                        'lunch': random.randint(30, 60),  # minutes
                        'short_breaks': random.randint(2, 4),
                        'break_duration': random.randint(10, 15)  # minutes per break
                    },
                    'overtime': {
                        'scheduled': random.choice([True, False]),
                        'hours': random.randint(1, 4) if random.random() < 0.2 else 0,
                        'reason': random.choice(['Staff Shortage', 'Emergency', 'Special Project', 'Training'])
                    },
                    'quality_metrics': {
                        'patient_satisfaction': random.uniform(3.5, 5.0),
                        'documentation_completion': random.uniform(0.8, 1.0),
                        'protocol_adherence': random.uniform(0.9, 1.0),
                        'team_collaboration': random.uniform(3.5, 5.0)
                    },
                    'notes': random.choice([
                        'Regular shift',
                        'Training day',
                        'Quality improvement focus',
                        'New protocol implementation',
                        'Team building activities'
                    ])
                }
                
                schedules.append(schedule)
                current_date += timedelta(days=1)
        
        return pd.DataFrame(schedules)

    def generate_equipment_maintenance(self) -> pd.DataFrame:
        """Generate comprehensive equipment maintenance data with enhanced hospital equipment scenarios."""
        maintenance_records = []
        
        # Get all equipment from hospital infrastructure
        equipment_list = []
        for facility, details in self.hospital_infrastructure['facilities'].items():
            if 'equipment' in details:
                if isinstance(details['equipment'], list):
                    for item in details['equipment']:
                        equipment_list.append({
                            'name': item,
                            'facility': facility,
                            'type': 'General'
                        })
                elif isinstance(details['equipment'], dict):
                    for category, items in details['equipment'].items():
                        if isinstance(items, list):
                            for item in items:
                                equipment_list.append({
                                    'name': item,
                                    'facility': facility,
                                    'type': category
                                })
                        else:
                            equipment_list.append({
                                'name': category,
                                'facility': facility,
                                'type': 'Specialized'
                            })
        
        # Generate maintenance records for each piece of equipment
        for equipment in equipment_list:
            # Generate multiple maintenance records for each equipment
            num_records = random.randint(1, 5)
            last_maintenance = datetime.now() - timedelta(days=random.randint(0, 365))
            
            for _ in range(num_records):
                maintenance_type = random.choice([
                    'Preventive Maintenance',
                    'Corrective Maintenance',
                    'Predictive Maintenance',
                    'Condition-Based Maintenance',
                    'Emergency Repair',
                    'Calibration',
                    'Software Update',
                    'Hardware Upgrade',
                    'Safety Inspection',
                    'Performance Testing'
                ])
                
                status = random.choice(['Scheduled', 'In Progress', 'Completed', 'Cancelled', 'Failed'])
                priority = random.choice(['Low', 'Medium', 'High', 'Critical'])
                
                if status == 'Completed':
                    completion_date = last_maintenance + timedelta(days=random.randint(1, 30))
                    duration = random.randint(1, 8)  # hours
                    cost = random.randint(100, 5000)
                    technician = self.fake.name()
                    parts_replaced = random.sample([
                        'Filter',
                        'Battery',
                        'Sensor',
                        'Circuit Board',
                        'Motor',
                        'Display',
                        'Keyboard',
                        'Power Supply',
                        'Cooling Fan',
                        'Cable'
                    ], k=random.randint(0, 3))
                else:
                    completion_date = None
                    duration = None
                    cost = None
                    technician = None
                    parts_replaced = []
                
                maintenance_record = {
                    'maintenance_id': f"MNT{random.randint(100000, 999999)}",
                    'equipment_id': f"EQP{random.randint(100000, 999999)}",
                    'equipment_name': equipment['name'],
                    'facility': equipment['facility'],
                    'equipment_type': equipment['type'],
                    'maintenance_type': maintenance_type,
                    'scheduled_date': last_maintenance,
                    'completion_date': completion_date,
                    'status': status,
                    'priority': priority,
                    'duration_hours': duration,
                    'cost': cost,
                    'technician': technician,
                    'parts_replaced': parts_replaced,
                    'maintenance_details': {
                        'description': random.choice([
                            'Routine maintenance check',
                            'Performance optimization',
                            'Safety compliance check',
                            'Emergency repair',
                            'System upgrade',
                            'Calibration adjustment',
                            'Preventive maintenance',
                            'Component replacement',
                            'Software update',
                            'Hardware upgrade'
                        ]),
                        'findings': random.choice([
                            'No issues found',
                            'Minor issues resolved',
                            'Major issues addressed',
                            'Critical issues fixed',
                            'Preventive measures implemented'
                        ]),
                        'recommendations': random.choice([
                            'Continue regular maintenance',
                            'Schedule follow-up inspection',
                            'Consider replacement',
                            'Upgrade recommended',
                            'No action needed'
                        ])
                    },
                    'quality_metrics': {
                        'downtime_hours': random.randint(0, 24) if status == 'Completed' else None,
                        'first_time_fix_rate': random.random(),
                        'mean_time_to_repair': random.randint(1, 48),
                        'preventive_maintenance_compliance': random.random(),
                        'equipment_reliability': random.random()
                    },
                    'safety_checks': {
                        'electrical_safety': random.choice(['Pass', 'Fail', 'N/A']),
                        'mechanical_safety': random.choice(['Pass', 'Fail', 'N/A']),
                        'radiation_safety': random.choice(['Pass', 'Fail', 'N/A']) if 'radiation' in equipment['name'].lower() else 'N/A',
                        'biological_safety': random.choice(['Pass', 'Fail', 'N/A']) if 'biological' in equipment['name'].lower() else 'N/A',
                        'chemical_safety': random.choice(['Pass', 'Fail', 'N/A']) if 'chemical' in equipment['name'].lower() else 'N/A'
                    },
                    'compliance': {
                        'regulatory_compliance': random.choice(['Compliant', 'Non-compliant', 'N/A']),
                        'certification_status': random.choice(['Certified', 'Not Certified', 'Pending']),
                        'inspection_date': last_maintenance - timedelta(days=random.randint(30, 180)),
                        'next_inspection_date': last_maintenance + timedelta(days=random.randint(180, 365))
                    },
                    'documentation': {
                        'work_order': f"WO{random.randint(100000, 999999)}",
                        'service_report': f"SR{random.randint(100000, 999999)}" if status == 'Completed' else None,
                        'warranty_status': random.choice(['Active', 'Expired', 'N/A']),
                        'maintenance_manual': random.choice(['Available', 'Not Available', 'N/A']),
                        'training_documentation': random.choice(['Complete', 'Incomplete', 'N/A'])
                    },
                    'cost_analysis': {
                        'labor_cost': random.randint(50, 500) if status == 'Completed' else None,
                        'parts_cost': random.randint(50, 2000) if status == 'Completed' else None,
                        'total_cost': cost,
                        'cost_justification': random.choice([
                            'Preventive maintenance',
                            'Emergency repair',
                            'Safety compliance',
                            'Performance improvement',
                            'Regulatory requirement'
                        ]) if status == 'Completed' else None
                    }
                }
                
                maintenance_records.append(maintenance_record)
                last_maintenance = completion_date if completion_date else last_maintenance
        
        return pd.DataFrame(maintenance_records)

    def generate_inventory_management(self) -> pd.DataFrame:
        """Generate comprehensive inventory management data with enhanced hospital inventory scenarios."""
        inventory_records = []
        
        # Define inventory categories and items
        inventory_categories = {
            'Medical Supplies': [
                'Bandages', 'Gauze', 'Syringes', 'Needles', 'Gloves', 'Masks', 'Catheters',
                'IV Bags', 'Tubing', 'Dressings', 'Sutures', 'Splints', 'Casts'
            ],
            'Pharmaceuticals': [
                'Antibiotics', 'Pain Medications', 'Anticoagulants', 'Insulin', 'Vaccines',
                'Chemotherapy Drugs', 'Anesthetics', 'Antivirals', 'Antifungals'
            ],
            'Laboratory Supplies': [
                'Test Tubes', 'Petri Dishes', 'Reagents', 'Culture Media', 'Microscopes',
                'Centrifuges', 'Pipettes', 'Lab Coats', 'Safety Goggles'
            ],
            'Surgical Equipment': [
                'Surgical Instruments', 'Sutures', 'Implants', 'Prosthetics', 'Surgical Masks',
                'Surgical Gowns', 'Sterile Drapes', 'Surgical Lights'
            ],
            'Diagnostic Equipment': [
                'X-ray Machines', 'MRI Machines', 'CT Scanners', 'Ultrasound Machines',
                'ECG Machines', 'Blood Pressure Monitors', 'Pulse Oximeters'
            ],
            'Patient Care Equipment': [
                'Hospital Beds', 'Wheelchairs', 'Walkers', 'Crutches', 'Patient Monitors',
                'Ventilators', 'Defibrillators', 'Infusion Pumps'
            ],
            'Emergency Equipment': [
                'Crash Carts', 'Defibrillators', 'Emergency Kits', 'Trauma Supplies',
                'Emergency Medications', 'Airway Management Kits'
            ],
            'Housekeeping Supplies': [
                'Cleaning Solutions', 'Disinfectants', 'Paper Products', 'Linens',
                'Waste Management Supplies', 'Personal Protective Equipment'
            ]
        }
        
        # Generate inventory records for each category and item
        for category, items in inventory_categories.items():
            for item in items:
                # Generate multiple inventory records for each item
                num_records = random.randint(1, 3)
                
                for _ in range(num_records):
                    current_stock = random.randint(0, 1000)
                    reorder_point = random.randint(50, 200)
                    reorder_quantity = random.randint(100, 500)
                    
                    inventory_record = {
                        'inventory_id': f"INV{random.randint(100000, 999999)}",
                        'item_id': f"ITM{random.randint(100000, 999999)}",
                        'category': category,
                        'item_name': item,
                        'current_stock': current_stock,
                        'unit': random.choice(['Each', 'Box', 'Case', 'Pack', 'Kit']),
                        'location': {
                            'building': random.choice(['Main', 'East', 'West', 'North', 'South']),
                            'floor': random.randint(1, 5),
                            'room': f"{random.randint(100, 999)}",
                            'shelf': random.choice(['A', 'B', 'C', 'D']),
                            'bin': random.randint(1, 50)
                        },
                        'supplier': {
                            'name': self.fake.company(),
                            'contact': self.fake.name(),
                            'phone': self.fake.phone_number(),
                            'email': self.fake.email(),
                            'contract_number': f"CNT{random.randint(100000, 999999)}"
                        },
                        'pricing': {
                            'unit_cost': round(random.uniform(1.0, 1000.0), 2),
                            'total_value': round(current_stock * random.uniform(1.0, 1000.0), 2),
                            'last_purchase_price': round(random.uniform(1.0, 1000.0), 2),
                            'price_history': [
                                {
                                    'date': (datetime.now() - timedelta(days=i*30)).strftime('%Y-%m-%d'),
                                    'price': round(random.uniform(1.0, 1000.0), 2)
                                }
                                for i in range(3)
                            ]
                        },
                        'inventory_management': {
                            'reorder_point': reorder_point,
                            'reorder_quantity': reorder_quantity,
                            'lead_time_days': random.randint(1, 30),
                            'safety_stock': random.randint(20, 100),
                            'max_stock': reorder_point + reorder_quantity,
                            'min_stock': reorder_point - reorder_quantity,
                            'stock_status': 'Low' if current_stock < reorder_point else 'Adequate',
                            'last_count_date': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
                            'next_count_date': (datetime.now() + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
                        },
                        'usage_metrics': {
                            'daily_usage': random.randint(1, 50),
                            'weekly_usage': random.randint(10, 200),
                            'monthly_usage': random.randint(50, 1000),
                            'usage_trend': random.choice(['Increasing', 'Decreasing', 'Stable']),
                            'seasonal_variation': random.choice(['High', 'Medium', 'Low', 'None'])
                        },
                        'quality_control': {
                            'expiration_date': (datetime.now() + timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
                            'lot_number': f"LOT{random.randint(10000, 99999)}",
                            'quality_status': random.choice(['Good', 'Quarantine', 'Disposal']),
                            'last_quality_check': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
                            'next_quality_check': (datetime.now() + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
                        },
                        'maintenance': {
                            'maintenance_required': random.random() < 0.3,
                            'last_maintenance': (datetime.now() - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d') if random.random() < 0.3 else None,
                            'next_maintenance': (datetime.now() + timedelta(days=random.randint(1, 180))).strftime('%Y-%m-%d') if random.random() < 0.3 else None,
                            'maintenance_status': random.choice(['Up to Date', 'Due', 'Overdue', 'N/A'])
                        },
                        'compliance': {
                            'regulatory_compliance': random.choice(['Compliant', 'Non-compliant', 'N/A']),
                            'certification_status': random.choice(['Certified', 'Not Certified', 'Pending']),
                            'inspection_date': (datetime.now() - timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d'),
                            'next_inspection_date': (datetime.now() + timedelta(days=random.randint(180, 365))).strftime('%Y-%m-%d')
                        },
                        'documentation': {
                            'msds_available': random.choice(['Yes', 'No', 'N/A']),
                            'instruction_manual': random.choice(['Available', 'Not Available', 'N/A']),
                            'warranty_status': random.choice(['Active', 'Expired', 'N/A']),
                            'training_documentation': random.choice(['Complete', 'Incomplete', 'N/A'])
                        }
                    }
                    
                    inventory_records.append(inventory_record)
        
        return pd.DataFrame(inventory_records)

    def generate_dataset(self):
        """Generate comprehensive healthcare dataset with enhanced hospital scenarios."""
        print("Generating QuantAI Hospital Dataset...")
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate patient demographics
        print("Generating patient demographics...")
        patient_df = self.generate_patient_demographics(num_patients=1000)
        
        # Generate medical history
        print("Generating medical history...")
        medical_history_df = self.generate_medical_history(patient_df)
        
        # Generate appointments
        print("Generating appointments...")
        appointments_df = self.generate_appointments(patient_df, num_appointments=2000)
        
        # Generate staff schedule
        print("Generating staff schedule...")
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() + timedelta(days=30)
        staff_schedule_df = self.generate_staff_schedule(start_date, end_date)
        
        # Generate equipment maintenance
        print("Generating equipment maintenance records...")
        equipment_maintenance_df = self.generate_equipment_maintenance()
        
        # Generate inventory management
        print("Generating inventory management records...")
        inventory_management_df = self.generate_inventory_management()
        
        # Combine all datasets
        datasets = {
            'patients': patient_df,
            'medical_history': medical_history_df,
            'appointments': appointments_df,
            'staff_schedule': staff_schedule_df,
            'equipment_maintenance': equipment_maintenance_df,
            'inventory_management': inventory_management_df
        }
        
        # Save datasets to CSV files
        print("Saving datasets to CSV files...")
        for name, df in datasets.items():
            filename = os.path.join(data_dir, f"quantai_hospital_{name}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")
            
        # Save hospital infrastructure
        infrastructure_file = os.path.join(data_dir, 'hospital_infrastructure.json')
        with open(infrastructure_file, 'w') as f:
            json.dump(self.hospital_infrastructure, f, indent=4)
        print(f"Saved {infrastructure_file}")
        
        print("Dataset generation completed successfully!")
        return datasets

    def _get_age_appropriate_conditions(self, age: int) -> List[str]:
        """Get a list of conditions appropriate for the given age."""
        conditions = []
        
        # Get all conditions from medical knowledge
        all_conditions = list(self.medical_knowledge['conditions'].keys())
        
        # Filter conditions based on age
        for condition in all_conditions:
            if condition in ['Hypertension', 'Type 2 Diabetes', 'Gastroesophageal Reflux Disease', 'Osteoarthritis', 'Chronic Back Pain']:
                if age >= 40:  # More common in older adults
                    conditions.append(condition)
            elif condition in ['Asthma', 'Allergic Rhinitis']:
                if age <= 50:  # More common in younger people
                    conditions.append(condition)
            elif condition in ['Anxiety', 'Depression', 'Migraine']:
                conditions.append(condition)  # Can occur at any age
            elif condition == 'Glaucoma':
                if age >= 60:  # More common in elderly
                    conditions.append(condition)
        
        # If no conditions are found, return some default conditions
        if not conditions:
            conditions = ['Hypertension', 'Type 2 Diabetes', 'Asthma', 'Anxiety', 'Depression']
            
        return conditions

if __name__ == "__main__":
    try:
        print("Starting dataset generation...")
        print("Initializing QuantAIHospitalDatasetGenerator...")
        generator = QuantAIHospitalDatasetGenerator()
        print("Generator initialized successfully")
        
        print("\nGenerating datasets...")
        print("1. Generating patient demographics...")
        patient_df = generator.generate_patient_demographics(num_patients=1000)
        print(f"Generated {len(patient_df)} patient records")
        
        print("\n2. Generating medical history...")
        medical_history_df = generator.generate_medical_history(patient_df)
        print(f"Generated {len(medical_history_df)} medical history records")
        
        print("\n3. Generating appointments...")
        appointments_df = generator.generate_appointments(patient_df, num_appointments=2000)
        print(f"Generated {len(appointments_df)} appointment records")
        
        print("\n4. Generating staff schedule...")
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() + timedelta(days=30)
        staff_schedule_df = generator.generate_staff_schedule(start_date, end_date)
        print(f"Generated {len(staff_schedule_df)} staff schedule records")
        
        print("\n5. Generating equipment maintenance records...")
        equipment_maintenance_df = generator.generate_equipment_maintenance()
        print(f"Generated {len(equipment_maintenance_df)} equipment maintenance records")
        
        print("\n6. Generating inventory management records...")
        inventory_management_df = generator.generate_inventory_management()
        print(f"Generated {len(inventory_management_df)} inventory management records")
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        print(f"\nSaving datasets to {data_dir}...")
        
        # Save datasets
        datasets = {
            'patients': patient_df,
            'medical_history': medical_history_df,
            'appointments': appointments_df,
            'staff_schedule': staff_schedule_df,
            'equipment_maintenance': equipment_maintenance_df,
            'inventory_management': inventory_management_df
        }
        
        for name, df in datasets.items():
            filename = os.path.join(data_dir, f"quantai_hospital_{name}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")
            
        # Save hospital infrastructure
        infrastructure_file = os.path.join(data_dir, 'hospital_infrastructure.json')
        with open(infrastructure_file, 'w') as f:
            json.dump(generator.hospital_infrastructure, f, indent=4)
        print(f"Saved {infrastructure_file}")
        
        print("\nDataset generation completed successfully!")
        
    except Exception as e:
        print(f"\nError generating dataset: {str(e)}")
        import traceback
        traceback.print_exc()