import numpy as np
import pandas as pd
import torch
import os
import json
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM
# Fallback for older versions of transformers
try:
    from transformers import GenerationConfig
except ImportError:
    # Define a dummy GenerationConfig if not available
    class GenerationConfig:
        def __init__(self, **kwargs):
            self.max_length = kwargs.get('max_length', 50)
            self.num_beams = kwargs.get('num_beams', 1)
            self.temperature = kwargs.get('temperature', 1.0)
            self.top_k = kwargs.get('top_k', 50)
            self.top_p = kwargs.get('top_p', 1.0)
            self.do_sample = kwargs.get('do_sample', False)
from sentence_transformers import SentenceTransformer
from bert_score import score
import textstat
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSelectionEnv(gym.Env):
    def __init__(self, 
                 questions_df,
                 all_skills=None, 
                 lstm_model_path="./models/dkt_model_pretrained_seq50.keras",
                 question_types=["fill_in_one", "multiple_choice", "algebra"],
                 max_seq_len=50,
                 max_steps=300, 
                 device='cpu',
                 w_answerability=50,
                 w_improvement=100,
                 w_coverage=0.5,
                 top_k=3,
                 weak_skills_threshold=0.4,
                 action_types=['type', 'skill'] 
                 ):
        super().__init__()
        self.action_types = action_types
        if all_skills is None:
            self.all_skills = [
                "* () positive reals",
                "-",
                "/",
                "Absolute Value",
                "Addition Whole Numbers",
                "Addition and Subtraction Fractions",
                "Addition and Subtraction Integers",
                "Addition and Subtraction Positive Decimals",
                "Algebraic Simplification",
                "Algebraic Solving",
                "Angles - Acute",
                "Angles - Obtuse",
                "Angles - Right",
                "Angles on Parallel Lines Cut by a Transversal",
                "Area Circle",
                "Area Irregular Figure",
                "Area Parallelogram",
                "Area Rectangle",
                "Area Trapezoid",
                "Area Triangle",
                "Box and Whisker",
                "Calculation with + - * /",
                "Calculations with Similar Figures",
                "Choose an Equation from Given Information",
                "Circle Graph",
                "Circumference",
                "Coefficient",
                "Combinatorics",
                "Combining Like Terms",
                "Commutative Property",
                "Complementary and Supplementary Angles",
                "Composition of Function Adding",
                "Compound Interest",
                "Computation with Real Numbers",
                "Congruence",
                "Conversion of Fraction Decimals Percents",
                "Counting Methods",
                "Definition Pi",
                "Distributive Property",
                "Divisibility Rules",
                "Division Fractions",
                "Division Whole Numbers",
                "Effect of Changing Dimensions of a Shape Proportionally",
                "English and Metric Terminology",
                "Equal As Balance Concept",
                "Equation Solving More Than Two Steps",
                "Equation Solving Two or Fewer Steps",
                "Equivalent Fractions",
                "Estimation",
                "Expanded",
                "Exponent",
                "Exponents",
                "Factoring Trinomials",
                "Finding Max and Min from a Quadratic Equation",
                "Finding Percents",
                "Finding Ratios",
                "Finding Slope From Equation",
                "Finding Slope From Situation",
                "Finding Slope from Graph",
                "Finding Slope from Ordered Pairs",
                "Finding fractions and ratios",
                "Finding y-intercept from Linear Equation",
                "Finding y-intercept from Linear Situation",
                "Fraction Of",
                "Geometric Definitions",
                "Graph Shape",
                "Graphing Inequalities on a number line",
                "Greatest Common Factor",
                "Histogram as Table or Graph",
                "Intercept",
                "Interior Angles Figures with More than 3 Sides",
                "Interior Angles Triangle",
                "Inverse Relations",
                "Least Common Multiple",
                "Line Plot",
                "Line Symmetry",
                "Line of Best-Fit",
                "Linear Equations",
                "Linear area volume conversion",
                "Mean",
                "Mean-Median-Mode-Range Differentiation",
                "Median",
                "Mode",
                "Monomial",
                "Multiplication Fractions",
                "Multiplication Whole Numbers",
                "Multiplication and Division Integers",
                "Multiplication and Division Positive Decimals",
                "Multiplying Monomials",
                "Multiplying non Monomial Polynomials",
                "Nets of 3D Figures",
                "Number Line",
                "Order of Operations",
                "Order of Operations All",
                "Ordering Fractions",
                "Ordering Integers",
                "Ordering Positive Decimals",
                "Ordering Real Numbers",
                "Ordering Whole Numbers",
                "Parallel and Perpendicular Lines",
                "Parallel and Perpendicular Slopes",
                "Parts of a Polyomial",
                "Pattern Finding",
                "Percent Discount",
                "Percent Increase or Decrease",
                "Percent Of",
                "Percents",
                "Perimeter of a Polygon",
                "Picking Equation and Inequality from Choices",
                "Point Plotting",
                "Polynomial Factors",
                "Prime Number",
                "Probability of Two Distinct Events",
                "Probability of a Single Event",
                "Properties and Classification Quadrilaterals",
                "Properties and Classification Rectangular Prisms",
                "Properties and Classification Triangles",
                "Properties of Numbers",
                "Proportion",
                "Pythagorean Theorem",
                "Quadratic Equation Solving",
                "Range",
                "Rate",
                "Reading a Ruler or Scale",
                "Recognize Linear Pattern",
                "Recognizing Equivalent Expressions",
                "Reflection",
                "Rotations",
                "Rounding",
                "Sampling Techniques",
                "Scale Factor",
                "Scatter Plot",
                "Scientific Notation",
                "Similar Figures",
                "Simplifying Expressions positive exponents",
                "Slope",
                "Solve Quadratic Equations Using Factoring",
                "Solving Inequalities",
                "Solving System of Equation",
                "Solving Systems of Linear Equations",
                "Solving for a variable",
                "Square Root",
                "Square Roots",
                "Standard and Word Notation",
                "Stem and Leaf Plot",
                "Substitution",
                "Subtraction Whole Numbers",
                "Surface Area Cylinder",
                "Surface Area Rectangular Prism",
                "Surface Area Sphere",
                "Surface Area of 3D Objects",
                "Symbolization",
                "Table",
                "Terms",
                "Transformation",
                "Translations",
                "Understanding Concept of Probabilities",
                "Unit Conversion Standard to Metric",
                "Unit Conversion Within a System",
                "Unit Rate",
                "Variable",
                "Venn Diagram",
                "Volume Cone",
                "Volume Cylinder",
                "Volume Prism",
                "Volume Pyramid",
                "Volume Rectangular Prism",
                "Volume Sphere",
                "Volume of 3D Objects",
                "Write Linear Equation from Graph",
                "Write Linear Equation from Ordered Pairs",
                "Write Linear Equation from Situation",
                "Writing Expression from Diagrams",
                "X-Y Graph Reading"
            ]
        else:
            self.all_skills = all_skills

        self.device = device
        self.max_seq_len = max_seq_len
        self.weak_skills_threshold = weak_skills_threshold
        self.question_types = question_types
        self.num_skills = len(self.all_skills)
        self.num_question_types = len(self.question_types)
        self.max_steps = max_steps
        self.current_step = 0

        # Reward weights as parameters
        self.w_answerability = w_answerability
        self.w_improvement = w_improvement
        self.w_coverage = w_coverage

        # Define action space: [skill_id, question_type_id]
        self.action_space = spaces.MultiDiscrete([self.num_skills, self.num_question_types])

        # student's last performance per skill (vector)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_skills,), dtype=np.float32)

        # Try loading .h5 version first (more reliable)
        h5_model_path = lstm_model_path.replace('.keras', '.h5')
        
        # List of model paths to try in order
        model_paths = [
            h5_model_path,  # Try .h5 version first
            lstm_model_path,  # Then try the original .keras path
            lstm_model_path.replace('_seq50', '')  # Try without _seq50 suffix
        ]
        
        model_loaded = False
        last_error = None
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"ℹ️ Model not found at: {model_path}")
                continue
                
            print(f"\nAttempting to load model from: {model_path}")
            try:
                self.student_model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'tf': tf,
                        'keras': tf.keras,
                        'backend': tf.keras.backend,
                        'layers': tf.keras.layers,
                        'models': tf.keras.models,
                        'utils': tf.keras.utils
                    },
                    compile=False  # We don't need compilation for inference
                )
                print(f"✓ Successfully loaded student model from {model_path}")
                
                # Detect the input layer name for this model
                if hasattr(self.student_model, 'input_names') and self.student_model.input_names:
                    self.model_input_name = self.student_model.input_names[0]
                    print(f"ℹ️ Model input layer name: {self.model_input_name}")
                else:
                    # Fallback to 'input_1' if we can't detect it
                    self.model_input_name = 'input_1'
                    print(f"ℹ️ Using default input layer name: {self.model_input_name}")
                
                model_loaded = True
                break
            except Exception as e:
                last_error = e
                print(f"❌ Error loading model from {model_path}")
                print(f"Error details: {str(e)}")
        
        if not model_loaded:
            print("\n❌ Failed to load any model. Tried the following paths:")
            for path in model_paths:
                exists = "(exists)" if os.path.exists(path) else "(not found)"
                print(f"  - {path} {exists}")
            print("\nPlease check that the model files are not corrupted and are compatible with TensorFlow", tf.__version__)
            if last_error:
                print(f"Last error details: {str(last_error)}")
            raise RuntimeError("Failed to load any student model")

        self.top_k = top_k  # Store top_k globally
        self.top_k_indices = None  # Will be set after question generation

        # Load BERT for semantic similarity
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.skill_embs = self.sbert.encode(self.all_skills) 
        
        # Load questions
        self.questions_df = questions_df

        self.history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0
        self.student_performance_history = []

        self.current_question_embedding = None 
        self.question_embedding_cache = {}  # Add cache for question embeddings


    def predict_student_performance(self, new_skill_id=None, new_question_type_id=None):
        """
        Predicts the student's performance using the LSTM student model.
        Stores and updates the student's interaction history for use as LSTM input.
        Decodes the skill_id to the same encoding used as LSTM input.
        
        Args:
            new_skill_id: If provided, adds this skill to the sequence for prediction
            new_question_type_id: If provided, adds this question type to the sequence for prediction
        """

        # Prepare the latest sequence for prediction (pad/truncate as needed)
        max_seq_len = self.max_seq_len  # or use the same as in training
        num_skills = self.num_skills

        if len(self.history) == 0 and new_skill_id is None:
            # If no history and no new skill, return NULL performance for all skills
            return np.ones(num_skills, dtype=np.float32) * 0

        # Only use relevant fields for encoding
        history_to_encode = [
            (h["skill_id"], h["predicted_correctness_for_skill"], h["question_type_id"])
            for h in self.history
            if "skill_id" in h and "predicted_correctness_for_skill" in h and "question_type_id" in h
        ]
        
        # Add new skill/question type if provided
        if new_skill_id is not None and new_question_type_id is not None:
            # Use current performance for this skill as assumption
            if hasattr(self, 'student_performance') and self.student_performance is not None:
                assumed_correct = self.student_performance[new_skill_id] 
            
            history_to_encode.append((new_skill_id, assumed_correct, new_question_type_id))

        # Create input sequence using compressed encoding
        x_seq = []
        for s, c, q in history_to_encode:
            # Map skill_id to a compressed range [0, 33]
            compressed_skill_id = s % 34  # 34 skills per segment
            segment_id = s // 34  # Which segment (0-4 for 174 skills)
            
            # Encode timestep: skill_id + segment*34 + correctness*num_segments*34 + qtype*2*num_segments*34
            feature_val = (compressed_skill_id + 
                         segment_id * 34 + 
                         int(c) * 5 * 34 +  # 5 segments
                         q * 2 * 5 * 34)    # 2 for binary correctness      
            
            x_seq.append(feature_val)
        
        # Pad sequence
        x_seq = pad_sequences([x_seq], maxlen=max_seq_len, padding='post')[0]
        
        # Create one-hot encoded matrix
        X_input = np.zeros((1, max_seq_len, 100))
        for i, val in enumerate(x_seq):
            if val == 0 and i >= len(history_to_encode):  # Skip padding
                continue
            # Convert to one-hot, ensuring index is within bounds
            idx = int(val) % 100  # Ensure index is within [0,99]
            X_input[0, i, idx] = 1

        # Predict with LSTM model - use the detected input name
        try:
            # Try using the detected input name (e.g., 'skill_input' or 'input_1')
            y_pred = self.student_model.predict({self.model_input_name: X_input}, verbose=0)
        except Exception as e:
            # Fallback: try passing the array directly without dictionary
            print(f"⚠️ Error with named input '{self.model_input_name}': {e}")
            print(f"⚠️ Attempting to predict with direct array input...")
            try:
                y_pred = self.student_model.predict(X_input, verbose=0)
            except Exception as e2:
                print(f"❌ Direct array prediction also failed: {e2}")
                # Last resort: return zero performance
                print(f"⚠️ Returning zero performance as fallback")
                self.student_performance = np.zeros(self.num_skills, dtype=np.float32)
                return self.student_performance

        # --- START OF FIX ---
        # Our working model (dkt_model_working.keras) returns a 2D tensor of shape (batch_size, 1),
        # not a 3D tensor of per-skill predictions.
        # We will take this single output value as a general mastery prediction.
        
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            # This handles the output of dkt_model_working.keras
            predicted_mastery_score = y_pred[0, 0]
            # Broadcast this single mastery score to all skills
            self.student_performance = np.full(self.num_skills, predicted_mastery_score, dtype=np.float32)
        elif y_pred.ndim == 3:
            # This handles a true DKT model that might be used in the future
            last_idx = min(len(history_to_encode) - 1, max_seq_len - 1)
            if last_idx < 0: last_idx = 0
            self.student_performance = y_pred[0, last_idx, :]
        else:
            # Fallback for unexpected model output shape
            print(f"Warning: Unexpected model output shape: {y_pred.shape}")
            self.student_performance = np.zeros(self.num_skills, dtype=np.float32)
        # --- END OF FIX ---
            
        return self.student_performance
    
    def get_skill_performance(self, skill_id):
        """
        Get the student's performance for a specific skill.
        This is a wrapper around get_skill_performance for clarity.
        """
        return self.student_performance[skill_id]

    def update_history(self, info):
        """
        Update the student's interaction history.
        """
        self.history.append(info)

    def weak_skill_coverage(self, selected_skill_id=None):
        """
        Measures how well the selected question targets the student's weakest skills,
        without using embeddings. Returns 1 if the skill is among the weakest, else 0.
        """
        # Calculate performance for each skill as a NumPy array
        skill_performances = np.array([self.get_skill_performance(skill_id) for skill_id in range(self.num_skills)], dtype=np.float32)

        # Identify the bottom X% of skills (most needed) based on threshold
        sorted_indices = np.argsort(skill_performances)
        bottom_percent = int(self.weak_skills_threshold * len(sorted_indices))
        weak_skills = set(sorted_indices[:bottom_percent])

        # If selected_skill_id is not provided, use last skill from history
        if selected_skill_id is None and self.history:
            selected_skill_id = self.history[-1]["skill_id"]

        # Coverage is 1 if selected skill is among the weakest, else 0
        coverage = 1.0 if selected_skill_id in weak_skills else 0.0
        return coverage

    # def weak_skill_coverage(self):
    #     """
    #     Analyzes student performance gaps across all skills and checks if the question
    #     addresses the most needed skills based on performance history using SBERT similarity.
    #     Uses NumPy arrays for performance-critical operations.
    #     """
    #     if not hasattr(self, "history") or not self.history:
    #         return 1.0

    #     # Calculate performance for each skill as a NumPy array
    #     skill_performances = np.array([self.get_skill_performance(skill_id) for skill_id in range(self.num_skills)], dtype=np.float32)

    #     # Identify the bottom X% of skills (most needed) based on threshold
    #     sorted_indices = np.argsort(skill_performances)
    #     bottom_percent = int(self.weak_skills_threshold * len(sorted_indices))
    #     weak_skills = sorted_indices[:bottom_percent]

    #     # Get embeddings for the question and weak skills
    #     question_embedding = self.current_question_embedding
    #     weak_skill_names = np.array([self.all_skills[skill_id] for skill_id in weak_skills])
    #     weak_skill_embeddings = self.sbert.encode(weak_skill_names)

    #     # Calculate cosine similarities using sklearn
    #     similarities = cosine_similarity(weak_skill_embeddings, question_embedding.reshape(1, -1)).flatten()

    #     # Weight similarities by how weak each skill is (lower performance = higher weight)
    #     weakness_weights = 1.0 - skill_performances[weak_skills]  # NumPy array
    #     weighted_similarities = similarities * weakness_weights    # Element-wise multiplication

    #     # Use average similarity as alignment score, normalized to [0, 1]
    #     alignment_score = np.clip(np.mean(weighted_similarities), 0, 1)

    #     return alignment_score


    def calculate_reward(self, improvement, answerability, coverage):
        improvement = self.w_improvement * improvement
        answerability = self.w_answerability * answerability
        coverage = self.w_coverage * coverage

        reward = (
            improvement + #tanh (-1 to 1)
            answerability + #sigmoid (0-1)
            coverage 
        )
        return reward

    def compute_answerability(self):
        """
        Estimate how answerable a question is based on the student's performance on top-k relevant skills.
        Uses self.top_k_indices set during question generation.
        """
        try:
            performances = [self.get_skill_performance(i) for i in self.top_k_indices]
            weights = np.ones(len(performances)) / (len(performances) + 1e-8)  # Uniform weights
            weighted_perf = np.dot(weights, performances)
            return weighted_perf
        except Exception as e:
            print(f"Error in compute_answerability: {e}")
            return 0.5

    def _select_question(self, skill_id, question_type_id):
        """
        Select a question based on skill_id and question_type_id.
        This is a public-facing method used by the API.
        Returns a pandas Series representing the selected question.
        """
        if skill_id < 0 or skill_id >= self.num_skills:
            raise ValueError(f"Skill id '{skill_id}' out of range.")
        if question_type_id < 0 or question_type_id >= self.num_question_types:
            raise ValueError(f"Question type id '{question_type_id}' out of range.")
        
        skill = self.all_skills[skill_id]
        question_type = self.question_types[question_type_id]
        
        return self.get_question_from_bank(skill, question_type)
    
    def get_question_from_bank(self, skill, question_type):
        # Filter for matching skill and question_type
        matches = self.questions_df[
            (self.questions_df['skill'] == skill) &
            (self.questions_df['question_type'] == question_type)
        ]
        if matches.empty:
            # Fallback: pick any question with the skill
            matches = self.questions_df[self.questions_df['skill'] == skill]
        if matches.empty:
            # Fallback: pick any question with the question_type
            matches = self.questions_df[self.questions_df['question_type'] == question_type]
        if matches.empty:
            # Fallback: pick any question
            matches = self.questions_df
        # Randomly select one
        question_row = matches.sample(1).iloc[0]
        question_text = question_row['question_text']

        # Cache question embedding
        if question_text not in self.question_embedding_cache:
            self.question_embedding_cache[question_text] = self.sbert.encode([question_text])[0]
        self.current_question_embedding = self.question_embedding_cache[question_text]

        # Compute cosine similarities and store top_k_indices
        sims = cosine_similarity(self.skill_embs, self.current_question_embedding.reshape(1, -1)).flatten()
        self.top_k_indices = np.argsort(sims)[-self.top_k:]

        return question_row

    def step(self, action):
        # Handle action_types logic
        if not self.action_types:  # Random selection
            skill_id = np.random.randint(self.num_skills)
            question_type_id = np.random.randint(self.num_question_types)
        elif self.action_types == ['skill']:
            skill_id = int(action[0])
            question_type_id = np.random.randint(self.num_question_types)
        elif self.action_types == ['type', 'skill'] or self.action_types == ['skill', 'type']:
            skill_id = int(action[0])
            question_type_id = int(action[1])
        else:
            raise ValueError(f"Unsupported action_types: {self.action_types}")

        if skill_id < 0 or skill_id >= self.num_skills:
            raise ValueError(f"Skill id '{skill_id}' out of range.")
        if question_type_id < 0 or question_type_id >= self.num_question_types:
            raise ValueError(f"Question type id '{question_type_id}' out of range.")

        skill = self.all_skills[skill_id]
        question_type = self.question_types[question_type_id]

        # Get question metadata from bank and normalize values
        question_row = self.get_question_from_bank(skill, question_type)
        raw_question_dict = question_row.to_dict() if hasattr(question_row, "to_dict") else dict(question_row)
        question_data = {}
        for key, value in raw_question_dict.items():
            if pd.isna(value):
                question_data[key] = None
            elif isinstance(value, np.generic):
                question_data[key] = value.item()
            else:
                question_data[key] = value

        question_text = question_data.get("question_text", "")
        difficulty = question_data.get("difficulty") or "unknown"
        
        student_performance = self.predict_student_performance(skill_id, question_type_id)
        # Store performance history
        self.student_performance_history.append(student_performance)

        # Use average performance improvement across all skills
        if len(self.student_performance_history) > 1:
            prev_avg_perf = float(np.mean(self.student_performance_history[-2]))
            curr_avg_perf = float(np.mean(self.student_performance_history[-1]))
            improvement_reward = curr_avg_perf - prev_avg_perf
        else:
            improvement_reward = 0.0

        answerability = self.compute_answerability()
        weak_skill_coverage = self.weak_skill_coverage()

        reward = self.calculate_reward(improvement_reward, answerability, weak_skill_coverage)

        # Observation: student's predicted performance for all skills
        observation = student_performance.astype(np.float32)
        info = {
            "question": question_text,
            "question_metadata": question_data,
            "skill": skill,
            "improvement": float(improvement_reward*self.w_improvement),
            "answerability": float(answerability*self.w_answerability),
            "coverage": float(weak_skill_coverage*self.w_coverage),
            "question_type": question_type,
            "difficulty": difficulty,
            "predicted_correctness_for_skill": float(student_performance[skill_id]),
            "student_performance_per_skill": {k: float(v) for k, v in zip(self.all_skills, student_performance.tolist())},
            "skill_id": int(skill_id),
            "question_type_id": int(question_type_id),
            "reward": float(reward),
        }


        # Update action history
        self.update_history(info)

        self.current_step += 1
        done = self.current_step >= self.max_steps  # <-- done when max_steps reached

        if done:
            self.save_history_json()

        print(
            f"Step {self.current_step} | "
            f"Question: {info['question']} | "
            f"Skill: {info['skill']} | "
            f"Difficulty: {info['difficulty']} | "
            f"Type: {info['question_type']} | "
            f"AvgPerf: {(np.mean(student_performance) * self.w_improvement):.3f} | "
            f"Improvement: {info['improvement']:.3f} | "
            f"Answerability: {info['answerability']:.3f} | "
            f"Coverage: {info['coverage']:.3f} | "
            f"Reward: {reward:.3f}"
        )
        # print(f'Step {self.current_step} Complete')

        # Return observation, reward, terminated, truncated, info (Gymnasium API)
        truncated = False  # We don't use truncation in this environment
        return observation, reward, done, truncated, info

    def save_history_json(self, path="trainings/history/history.json"):
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2, default=default_serializer)

    def reset(self, seed=None, options=None):
        """
        Reset the environment and student history.
        Returns initial observation and info dict (Gymnasium API).
        """
        super().reset(seed=seed)
        
        self.history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0
        self.current_step = 0 
        self.current_question_embedding = None
    
        # Reset student performance history
        self.student_performance_history = []

        # Return neutral performance for all skills and empty info dict
        observation = np.ones(self.num_skills, dtype=np.float32) * 0
        info = {}
        return observation, info

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self, "action_space"):
            self.action_space.seed(seed)
        if hasattr(self, "observation_space"):
            self.observation_space.seed(seed)
        return [seed]