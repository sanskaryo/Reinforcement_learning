#       QUIZ COMPLETE!
# ======================================================================

# 📊 Final Score: 5/10 (50%)

# 🎯 Skills You Practiced:
#    🟡 Conversion of Fraction Decimals Percents      | Mastery: 55%
#    🟡 Rate                                          | Mastery: 47%
#    🟡 Sampling Techniques                           | Mastery: 51%
#    🟡 Terms                                         | Mastery: 60%

# 📈 Overall Average Mastery: 53%

# 💡 Recommended Topics to Practice:
#                     QUIZ COMPLETE!
# ======================================================================

# 📊 Final Score: 5/10 (50%)

# 🎯 Skills You Practiced:
#    🟡 Conversion of Fraction Decimals Percents      | Mastery: 55%
#    🟡 Rate                                          | Mastery: 47%
#    🟡 Sampling Techniques                           | Mastery: 51%
#    🟡 Terms                                         | Mastery: 60%

# 📈 Overall Average Mastery: 53%

# 💡 Recommended Topics to Practice:
# 📊 Final Score: 5/10 (50%)

# 🎯 Skills You Practiced:
#    🟡 Conversion of Fraction Decimals Percents      | Mastery: 55%
#    🟡 Rate                                          | Mastery: 47%
#    🟡 Sampling Techniques                           | Mastery: 51%
#    🟡 Terms                                         | Mastery: 60%

# 📈 Overall Average Mastery: 53%

# 💡 Recommended Topics to Practice:
#    🟡 Conversion of Fraction Decimals Percents      | Mastery: 55%
#    🟡 Rate                                          | Mastery: 47%
#    🟡 Sampling Techniques                           | Mastery: 51%
#    🟡 Terms                                         | Mastery: 60%

# 📈 Overall Average Mastery: 53%

# 💡 Recommended Topics to Practice:
#    🟡 Sampling Techniques                           | Mastery: 51%
#    🟡 Terms                                         | Mastery: 60%

# 📈 Overall Average Mastery: 53%

# 💡 Recommended Topics to Practice:
#    🟡 Terms                                         | Mastery: 60%

# 📈 Overall Average Mastery: 53%

# 💡 Recommended Topics to Practice:
# 💡 Recommended Topics to Practice:
#    • Rate (Current: 47%)

# ======================================================================

# 💾 Your session has been saved to: quiz_session_history.json


# (venv_rl) C:\Sankhu Codes and Stuff\Others\Other Projects\Revolutionizing-EdTech-Through-AI-main\rl-proj\personalized_learning_rl>


















"""
Simple Adaptive Math Quiz - No Model Required
This version works without loading pre-trained models
Users answer questions and difficulty adapts based on performance
"""

import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_welcome():
    """Display welcome message"""
    clear_screen()
    print("="*70)
    print("        ADAPTIVE MATH QUIZ - Personalized Learning System")
    print("="*70)
    print()
    print("This quiz adapts to YOUR learning level!")
    print("• Answer questions to the best of your ability")
    print("• The system will adjust difficulty based on your performance")
    print("• Track your progress across different math skills")
    print()
    input("Press ENTER to start...")

def display_question(question_num, total_questions, question_info):
    """Display a single question"""
    clear_screen()
    print("="*70)
    print(f"Question {question_num}/{total_questions}")
    print("="*70)
    print()
    print(f"📚 Topic: {question_info['skill']}")
    print(f"📊 Difficulty: {question_info['difficulty']}")
    print()
    print(f"❓ {question_info['question_text']}")
    print()
    
    # Display choices if multiple choice
    if 'choices' in question_info and pd.notna(question_info['choices']) and question_info['choices']:
        choices = str(question_info['choices']).split('|')
        print("Choose from:")
        for i, choice in enumerate(choices, 1):
            print(f"   {i}. {choice}")
        print()

def get_user_answer(question_info):
    """Get answer from user"""
    if 'choices' in question_info and pd.notna(question_info['choices']) and question_info['choices']:
        choices = str(question_info['choices']).split('|')
        while True:
            try:
                choice_num = input("Your choice (enter number): ")
                if choice_num.isdigit() and 1 <= int(choice_num) <= len(choices):
                    return choices[int(choice_num) - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        return input("Your answer: ")

def check_answer(user_answer, correct_answer):
    """Check if answer is correct"""
    # Normalize answers for comparison
    user = str(user_answer).strip().lower().replace(" ", "").replace("−", "-")
    correct = str(correct_answer).strip().lower().replace(" ", "").replace("−", "-")
    
    # Handle various formats
    return user == correct or user in correct or correct in user

def display_feedback(is_correct, correct_answer, topic, mastery):
    """Display feedback to user"""
    print()
    print("-"*70)
    if is_correct:
        print("✅ CORRECT! Well done!")
    else:
        print(f"❌ Not quite. The correct answer was: {correct_answer}")
    
    print()
    print(f"Your current mastery of '{topic}': {mastery:.0%}")
    print("-"*70)
    print()
    input("Press ENTER to continue...")

def select_next_question(questions_df, skill_mastery, last_skill=None):
    """
    Adaptively select the next question based on student performance
    """
    # Find weakest skills (lowest mastery)
    weak_skills = sorted(skill_mastery.items(), key=lambda x: x[1])[:3]
    
    # Prefer weak skills, but vary for coverage
    if weak_skills and np.random.random() > 0.3:
        selected_skill = weak_skills[0][0]
    else:
        # Occasionally pick a random skill for variety
        unique_skills = questions_df['skill'].unique()
        selected_skill = np.random.choice(unique_skills)
    
    # Filter by skill
    skill_questions = questions_df[questions_df['skill'] == selected_skill]
    
    if len(skill_questions) == 0:
        # If no questions for that skill, pick any random question
        return questions_df.sample(1).iloc[0]
    
    # Determine difficulty based on mastery
    mastery = skill_mastery.get(selected_skill, 0.5)
    
    if mastery < 0.3:
        # Very weak - focus on easy questions
        difficulty_questions = skill_questions[skill_questions['difficulty'] == 'easy']
        if len(difficulty_questions) == 0:
            difficulty_questions = skill_questions
    elif mastery < 0.6:
        # Weak - mix of easy and medium
        difficulty_questions = skill_questions[
            (skill_questions['difficulty'] == 'easy') | 
            (skill_questions['difficulty'] == 'medium')
        ]
        if len(difficulty_questions) == 0:
            difficulty_questions = skill_questions
    else:
        # Strong - include all difficulties
        difficulty_questions = skill_questions
    
    # Select random question from filtered set
    return difficulty_questions.sample(1).iloc[0]

def display_progress(stats, skill_mastery):
    """Display current progress"""
    print()
    print("📊 Your Progress So Far:")
    print(f"   Correct: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.0f}%)")
    print(f"   Topics covered: {len(skill_mastery)}")
    print(f"   Average mastery: {np.mean(list(skill_mastery.values())):.0%}")
    print()

def display_final_summary(stats, skill_mastery):
    """Display final summary"""
    clear_screen()
    print("="*70)
    print("                    QUIZ COMPLETE!")
    print("="*70)
    print()
    print(f"📊 Final Score: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.0f}%)")
    print()
    
    print("🎯 Skills You Practiced:")
    for skill in sorted(skill_mastery.keys()):
        mastery = skill_mastery[skill]
        emoji = "🟢" if mastery > 0.7 else "🟡" if mastery > 0.4 else "🔴"
        print(f"   {emoji} {skill[:45]:45} | Mastery: {mastery:.0%}")
    print()
    
    # Overall improvement
    avg_mastery = np.mean(list(skill_mastery.values()))
    print(f"📈 Overall Average Mastery: {avg_mastery:.0%}")
    print()
    
    # Recommendations
    weak_skills = [(skill, mastery) for skill, mastery in skill_mastery.items() if mastery < 0.5]
    if weak_skills:
        print("💡 Recommended Topics to Practice:")
        for skill, mastery in sorted(weak_skills, key=lambda x: x[1])[:5]:
            print(f"   • {skill} (Current: {mastery:.0%})")
    else:
        print("🌟 Excellent work! You're doing great across all topics!")
    
    print()
    print("="*70)

def run_interactive_quiz(num_questions=10):
    """
    Main function to run interactive quiz
    
    Args:
        num_questions: Number of questions to ask (default: 10)
    """
    # Welcome
    display_welcome()
    
    # Load questions
    print("Loading quiz system...")
    try:
        questions_df = pd.read_csv("./data/questions.csv")
        print(f"✓ Loaded {len(questions_df)} questions")
    except FileNotFoundError:
        print("❌ Error: questions.csv not found in ./data/")
        print("Please ensure you're running from the correct directory")
        return
    
    # Initialize tracking
    stats = {
        'correct': 0,
        'total': 0,
    }
    skill_mastery = defaultdict(lambda: 0.5)  # Initialize at 50% mastery
    history = []
    
    print("✓ Ready!")
    print()
    input("Press ENTER to begin...")
    
    # Quiz loop
    for question_num in range(1, num_questions + 1):
        # Select next question adaptively
        question_info = select_next_question(questions_df, skill_mastery)
        skill = question_info['skill']
        
        # Display question
        display_question(question_num, num_questions, question_info)
        
        # Get user answer
        user_answer = get_user_answer(question_info)
        
        # Check answer
        is_correct = check_answer(user_answer, question_info['answer'])
        
        # Update stats
        stats['total'] += 1
        if is_correct:
            stats['correct'] += 1
        
        # Update skill mastery using simple weighted average
        old_mastery = skill_mastery[skill]
        if is_correct:
            # Increase mastery by 10% if correct
            new_mastery = old_mastery + (1 - old_mastery) * 0.1
        else:
            # Decrease mastery by 5% if incorrect
            new_mastery = old_mastery * 0.95
        
        skill_mastery[skill] = new_mastery
        
        # Store in history
        history.append({
            'question_num': question_num,
            'skill': skill,
            'is_correct': is_correct,
            'mastery': new_mastery,
            'difficulty': question_info['difficulty']
        })
        
        # Show feedback
        display_feedback(
            is_correct, 
            question_info['answer'],
            skill,
            new_mastery
        )
        
        # Show progress every 3 questions (if not the last question)
        if question_num % 3 == 0 and question_num < num_questions:
            display_progress(stats, skill_mastery)
            input("Press ENTER to continue...")
    
    # Final summary
    display_final_summary(stats, skill_mastery)
    
    # Save history
    history_file = "quiz_session_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Your session has been saved to: {history_file}")
    print()

if __name__ == "__main__":
    import sys
    
    # Get number of questions from command line or use default
    if len(sys.argv) > 1:
        try:
            num_q = int(sys.argv[1])
        except ValueError:
            num_q = 10
            print(f"Invalid number, using default: {num_q} questions")
    else:
        num_q = 10
    
    print(f"Starting quiz with {num_q} questions...")
    print()
    
    run_interactive_quiz(num_questions=num_q)
