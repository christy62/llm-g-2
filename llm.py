import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer

class HelixPredictor:
    """Wrapper for trained T5 helix reconstruction model."""
    
    def __init__(self, model_path="./t5_helix_model_final"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, hits_string):
        """Predict vertex and momentum from hits string."""
        
        inputs = self.tokenizer(
            hits_string,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        prediction = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        
        parsed = self._parse_gen_event(prediction)
        
        if parsed:
            parsed['prediction_string'] = prediction
            return parsed
        else:
            return {'prediction_string': prediction, 'error': 'Failed to parse'}
    
    def _parse_gen_event(self, prediction):
        try:
            if not prediction.startswith("gen_event("):
                return None
            
            content = prediction[len("gen_event("):-1]
            parts = content.split(", mom=")
            
            if len(parts) != 2:
                return None
            
            vtx_str = parts[0].replace("vtx=", "").strip("[]")
            mom_str = parts[1].strip("[]")
            
            vertex = [float(x) for x in vtx_str.split(",")]
            momentum = [float(x) for x in mom_str.split(",")]
            
            return {'vertex': vertex, 'momentum': momentum}
        except Exception:
            return None
    
    # ==================== EVALUATION FUNCTIONS ====================
    # These compute errors WITHOUT affecting model weights
    
    def evaluate_token_loss(self, input_str, target_str):
        """
        Compute token-level cross-entropy loss for evaluation purposes only.
        No gradients, no weight updates.
        
        Args:
            input_str: Input hits string
            target_str: Target gen_event string
            
        Returns:
            Dictionary with loss value
        """
        # Tokenize input and target
        inputs = self.tokenizer(
            input_str,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        targets = self.tokenizer(
            target_str,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=targets['input_ids']
            )
        
        return {
            'token_loss': outputs.loss.item(),
            'num_tokens': targets['input_ids'].shape[1]
        }
    
    def evaluate_physics_errors(self, prediction_parsed, truth_parsed):
        """
        Compute physics-aware errors between prediction and truth.
        
        Args:
            prediction_parsed: Dictionary from _parse_gen_event() or predict()
            truth_parsed: Dictionary with 'vertex' and 'momentum' lists
            
        Returns:
            Dictionary with various error metrics
        """
        if 'vertex' not in prediction_parsed or 'momentum' not in prediction_parsed:
            return {'error': 'Invalid prediction format'}
        
        # Convert to numpy arrays
        pred_vtx = np.array(prediction_parsed['vertex'])
        pred_mom = np.array(prediction_parsed['momentum'])
        
        true_vtx = np.array(truth_parsed['vertex'])
        true_mom = np.array(truth_parsed['momentum'])
        
        # Vertex errors
        vtx_abs_error = np.linalg.norm(pred_vtx - true_vtx)
        vtx_x_error = abs(pred_vtx[0] - true_vtx[0])
        vtx_y_error = abs(pred_vtx[1] - true_vtx[1])
        vtx_z_error = abs(pred_vtx[2] - true_vtx[2])
        
        # Momentum errors
        mom_abs_error = np.linalg.norm(pred_mom - true_mom)
        mom_rel_error = mom_abs_error / (np.linalg.norm(true_mom) + 1e-8)  # avoid division by zero
        
        mom_px_error = abs(pred_mom[0] - true_mom[0])
        mom_py_error = abs(pred_mom[1] - true_mom[1])
        mom_pz_error = abs(pred_mom[2] - true_mom[2])
        
        return {
            # Vertex errors
            'vertex_absolute_error': float(vtx_abs_error),
            'vertex_x_error': float(vtx_x_error),
            'vertex_y_error': float(vtx_y_error),
            'vertex_z_error': float(vtx_z_error),
            
            # Momentum errors
            'momentum_absolute_error': float(mom_abs_error),
            'momentum_relative_error': float(mom_rel_error),
            'momentum_px_error': float(mom_px_error),
            'momentum_py_error': float(mom_py_error),
            'momentum_pz_error': float(mom_pz_error),
            
            # Combined metrics
            'total_physics_error': float(vtx_abs_error + 0.1 * mom_abs_error)  # weighted combination
        }
    
    def evaluate_single_example(self, input_str, target_str):
        """
        Complete evaluation of a single example.
        
        Args:
            input_str: Input hits string
            target_str: Target gen_event string
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Make prediction
        prediction_result = self.predict(input_str)
        
        # Parse target
        truth_parsed = self._parse_gen_event(target_str)
        
        if not truth_parsed:
            return {'error': 'Could not parse target string'}
        
        if 'error' in prediction_result:
            return {'error': prediction_result['error']}
        
        # Get all metrics
        token_loss_result = self.evaluate_token_loss(input_str, target_str)
        physics_errors = self.evaluate_physics_errors(prediction_result, truth_parsed)
        
        # Combine all results
        return {
            'prediction': prediction_result['prediction_string'],
            'truth': target_str,
            'parsed_prediction': {
                'vertex': prediction_result['vertex'],
                'momentum': prediction_result['momentum']
            },
            'parsed_truth': {
                'vertex': truth_parsed['vertex'],
                'momentum': truth_parsed['momentum']
            },
            'token_loss': token_loss_result['token_loss'],
            'physics_errors': physics_errors,
            'format_correct': prediction_result['prediction_string'].startswith("gen_event(")
        }


# ==================== SEPARATE EVALUATION UTILITIES ====================

def batch_evaluate(predictor, examples):
    """
    Evaluate multiple examples and compute aggregate statistics.
    """
    all_results = []
    
    for i, example in enumerate(examples, start=1):
        result = predictor.evaluate_single_example(
            example['input'],
            example['target']
        )
        
        if 'error' not in result:
            all_results.append(result)

        # ---- PROGRESS PRINT ----
        if i % 10000 == 0:  
            print(f"Processed {i} examples", flush=True)
    
    if not all_results:
        return {'error': 'No valid results'}
    
    # Compute aggregate statistics
    token_losses = [r['token_loss'] for r in all_results]
    vertex_errors = [r['physics_errors']['vertex_absolute_error'] for r in all_results]
    momentum_errors = [r['physics_errors']['momentum_absolute_error'] for r in all_results]
    format_accuracy = sum(1 for r in all_results if r['format_correct']) / len(all_results)
    
    return {
        'num_examples': len(all_results),
        'format_accuracy': format_accuracy,
        'token_loss': {
            'mean': float(np.mean(token_losses)),
            'std': float(np.std(token_losses)),
            'min': float(np.min(token_losses)),
            'max': float(np.max(token_losses))
        },
        'vertex_error': {
            'mean': float(np.mean(vertex_errors)),
            'std': float(np.std(vertex_errors)),
            'min': float(np.min(vertex_errors)),
            'max': float(np.max(vertex_errors))
        },
        'momentum_error': {
            'mean': float(np.mean(momentum_errors)),
            'std': float(np.std(momentum_errors)),
            'min': float(np.min(momentum_errors)),
            'max': float(np.max(momentum_errors))
        },
        'per_example_results': all_results
    }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize predictor
    predictor = HelixPredictor()
    
    # Example 1: Pure inference (no loss, no truth)
    print("=== INFERENCE EXAMPLE ===")
    example_hits = "hits=[0.15,-0.23,0.31];[-0.08,0.12,0.45];[0.22,-0.17,0.28]"
    result = predictor.predict(example_hits)
    print(f"Input: {example_hits[:50]}...")
    print(f"Prediction: {result.get('prediction_string', 'ERROR')}")
    if 'vertex' in result:
        print(f"Vertex: {result['vertex']}")
        print(f"Momentum: {result['momentum']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Evaluation with ground truth
    print("=== EVALUATION EXAMPLE ===")
    example_input = "hits=[0.15,-0.23,0.31];[-0.08,0.12,0.45];[0.22,-0.17,0.28]"
    example_target = "gen_event(vtx=[0.05,-0.03,0.12], mom=[1.23,-0.56,2.01])"
    
    eval_result = predictor.evaluate_single_example(example_input, example_target)
    
    if 'error' not in eval_result:
        print(f"Token Loss: {eval_result['token_loss']:.4f}")
        print(f"Vertex Error: {eval_result['physics_errors']['vertex_absolute_error']:.6f}")
        print(f"Momentum Error: {eval_result['physics_errors']['momentum_absolute_error']:.6f}")
        print(f"Format Correct: {eval_result['format_correct']}")
    else:
        print(f"Evaluation Error: {eval_result['error']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Batch evaluation
    print("=== BATCH EVALUATION EXAMPLE ===")
    batch_examples = [
        {'input': example_input, 'target': example_target},
        {'input': example_input, 'target': "gen_event(vtx=[0.04,-0.02,0.11], mom=[1.20,-0.55,2.00])"}
    ]
    
    batch_results = batch_evaluate(predictor, batch_examples)
    if 'error' not in batch_results:
        print(f"Evaluated {batch_results['num_examples']} examples")
        print(f"Format Accuracy: {batch_results['format_accuracy']:.2%}")
        print(f"Avg Vertex Error: {batch_results['vertex_error']['mean']:.6f}")
        print(f"Avg Momentum Error: {batch_results['momentum_error']['mean']:.6f}")