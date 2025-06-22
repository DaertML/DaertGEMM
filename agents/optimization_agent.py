from ollama import chat
from ollama import ChatResponse

class OptimizationAgent:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name

    def suggest_primitives(self, code, hardware_description, hardware_hint, optimization_methods):
        # Placeholder for actual logic to suggest optimization primitives
        suggestions = []
        for method_name, method_prompt in optimization_methods.items():
            response = chat(model=self.model_name, messages=[{
                'role': 'user',
                'content': f"Suggest an optimization for the following code using the method '{method_name}':\n{code}\nHardware Description:\n{hardware_description}\nHardware Hint:\n{hardware_hint}\nOptimization Method:\n{method_prompt}",
            }])
            suggestion = response['message']['content']
            suggestions.append(suggestion)
        return suggestions

    def generate_code(self, optimization_method, base_code):
        # Placeholder for actual logic to generate optimized code
        response = chat(model=self.model_name, messages=[{
            'role': 'user',
            'content': f"Apply the '{optimization_method}' optimization method to the following code:\n{base_code}",
        }])
        optimized_code = response['message']['content']
        return optimized_code
