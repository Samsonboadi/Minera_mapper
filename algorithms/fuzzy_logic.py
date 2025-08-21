"""
Fuzzy logic algorithms for geological analysis and uncertainty modeling
"""

import numpy as np
import json
import os

# Optional imports with fallbacks
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class FuzzyLogicProcessor:
    """Fuzzy logic system for geological data analysis"""
    
    def __init__(self):
        self.fuzzy_variables = {}
        self.fuzzy_rules = []
        self.control_system = None
        self.simulation = None
        self.membership_functions = {}
        
    def create_fuzzy_variable(self, name, universe_range, variable_type='antecedent'):
        """Create a fuzzy variable with membership functions"""
        if variable_type == 'antecedent':
            var = ctrl.Antecedent(np.arange(universe_range[0], universe_range[1] + 0.01, 0.01), name)
        else:  # consequent
            var = ctrl.Consequent(np.arange(universe_range[0], universe_range[1] + 0.01, 0.01), name)
        
        # Define default membership functions
        if name.lower() in ['geological', 'lithology']:
            var['unfavorable'] = fuzz.trimf(var.universe, [0, 0, 0.3])
            var['moderately_favorable'] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
            var['highly_favorable'] = fuzz.trimf(var.universe, [0.7, 1, 1])
            
        elif name.lower() in ['magnetic', 'aeromagnetic']:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['moderate'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])
            
        elif name.lower() in ['radiometric', 'gamma']:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.35])
            var['moderate'] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
            var['high'] = fuzz.trimf(var.universe, [0.65, 1, 1])
            
        elif name.lower() in ['mineral', 'alteration']:
            var['absent'] = fuzz.trimf(var.universe, [0, 0, 0.2])
            var['weak'] = fuzz.trimf(var.universe, [0.1, 0.3, 0.5])
            var['moderate'] = fuzz.trimf(var.universe, [0.4, 0.6, 0.8])
            var['strong'] = fuzz.trimf(var.universe, [0.7, 1, 1])
            
        elif name.lower() in ['structural', 'lineament']:
            var['low_density'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['moderate_density'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high_density'] = fuzz.trimf(var.universe, [0.6, 1, 1])
            
        elif name.lower() in ['prospectivity', 'potential']:
            var['very_low'] = fuzz.trimf(var.universe, [0, 0, 0.2])
            var['low'] = fuzz.trimf(var.universe, [0.1, 0.25, 0.4])
            var['moderate'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.6, 0.8, 0.9])
            var['very_high'] = fuzz.trimf(var.universe, [0.8, 1, 1])
            
        else:
            # Default three-level membership functions
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
            var['medium'] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
            var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])
        
        self.fuzzy_variables[name] = var
        return var
    
    def create_custom_membership_function(self, variable_name, mf_name, mf_type, parameters):
        """Create custom membership function"""
        if variable_name not in self.fuzzy_variables:
            raise ValueError(f"Variable {variable_name} does not exist")
        
        var = self.fuzzy_variables[variable_name]
        
        if mf_type == 'triangular':
            var[mf_name] = fuzz.trimf(var.universe, parameters)
        elif mf_type == 'trapezoidal':
            var[mf_name] = fuzz.trapmf(var.universe, parameters)
        elif mf_type == 'gaussian':
            var[mf_name] = fuzz.gaussmf(var.universe, parameters[0], parameters[1])
        elif mf_type == 'sigmoid':
            var[mf_name] = fuzz.sigmf(var.universe, parameters[0], parameters[1])
        else:
            raise ValueError(f"Unsupported membership function type: {mf_type}")
    
    def add_fuzzy_rule(self, antecedents, consequent, rule_name=None):
        """Add a fuzzy rule to the system"""
        # Parse antecedents and consequent
        rule = ctrl.Rule(antecedents, consequent, label=rule_name)
        self.fuzzy_rules.append(rule)
        return rule
    
    def create_mineral_prospectivity_system(self):
        """Create a comprehensive fuzzy system for mineral prospectivity"""
        # Create input variables
        geological = self.create_fuzzy_variable('geological', [0, 1], 'antecedent')
        magnetic = self.create_fuzzy_variable('magnetic', [0, 1], 'antecedent')
        radiometric = self.create_fuzzy_variable('radiometric', [0, 1], 'antecedent')
        structural = self.create_fuzzy_variable('structural', [0, 1], 'antecedent')
        alteration = self.create_fuzzy_variable('alteration', [0, 1], 'antecedent')
        
        # Create output variable
        prospectivity = self.create_fuzzy_variable('prospectivity', [0, 1], 'consequent')
        
        # Define comprehensive rule set
        rules = [
            # Very high prospectivity rules
            ctrl.Rule(geological['highly_favorable'] & magnetic['high'] & 
                     radiometric['high'] & structural['high_density'] & 
                     alteration['strong'], prospectivity['very_high']),
            
            ctrl.Rule(geological['highly_favorable'] & magnetic['high'] & 
                     alteration['strong'], prospectivity['very_high']),
            
            # High prospectivity rules
            ctrl.Rule(geological['highly_favorable'] & magnetic['high'] & 
                     radiometric['moderate'], prospectivity['high']),
            
            ctrl.Rule(geological['moderately_favorable'] & magnetic['high'] & 
                     radiometric['high'] & structural['high_density'], prospectivity['high']),
            
            ctrl.Rule(geological['highly_favorable'] & structural['high_density'] & 
                     alteration['strong'], prospectivity['high']),
            
            # Moderate prospectivity rules
            ctrl.Rule(geological['moderately_favorable'] & magnetic['moderate'] & 
                     radiometric['moderate'], prospectivity['moderate']),
            
            ctrl.Rule(geological['highly_favorable'] & magnetic['low'] & 
                     radiometric['high'], prospectivity['moderate']),
            
            ctrl.Rule(geological['moderately_favorable'] & structural['moderate_density'] & 
                     alteration['moderate'], prospectivity['moderate']),
            
            # Low prospectivity rules
            ctrl.Rule(geological['moderately_favorable'] & magnetic['low'] & 
                     radiometric['low'], prospectivity['low']),
            
            ctrl.Rule(geological['unfavorable'] & magnetic['moderate'], prospectivity['low']),
            
            # Very low prospectivity rules
            ctrl.Rule(geological['unfavorable'] & magnetic['low'] & 
                     radiometric['low'], prospectivity['very_low']),
            
            ctrl.Rule(geological['unfavorable'] & structural['low_density'] & 
                     alteration['absent'], prospectivity['very_low'])
        ]
        
        self.fuzzy_rules = rules
        
        # Create control system
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        return self.control_system
    
    def create_gold_prospectivity_system(self):
        """Create fuzzy system specifically for gold prospectivity"""
        # Create variables with gold-specific membership functions
        geology = self.create_fuzzy_variable('geology', [0, 1], 'antecedent')
        geology['greenstone'] = fuzz.trimf(geology.universe, [0.7, 0.9, 1.0])
        geology['granite'] = fuzz.trimf(geology.universe, [0.5, 0.7, 0.9])
        geology['sedimentary'] = fuzz.trimf(geology.universe, [0.0, 0.2, 0.4])
        
        magnetic = self.create_fuzzy_variable('magnetic', [0, 1], 'antecedent')
        magnetic['low_anomaly'] = fuzz.trimf(magnetic.universe, [0.0, 0.1, 0.3])
        magnetic['moderate_anomaly'] = fuzz.trimf(magnetic.universe, [0.2, 0.5, 0.8])
        magnetic['high_anomaly'] = fuzz.trimf(magnetic.universe, [0.7, 0.9, 1.0])
        
        geochemistry = self.create_fuzzy_variable('geochemistry', [0, 1], 'antecedent')
        geochemistry['pathfinder_low'] = fuzz.trimf(geochemistry.universe, [0.0, 0.0, 0.3])
        geochemistry['pathfinder_moderate'] = fuzz.trimf(geochemistry.universe, [0.2, 0.5, 0.8])
        geochemistry['pathfinder_high'] = fuzz.trimf(geochemistry.universe, [0.7, 1.0, 1.0])
        
        alteration = self.create_fuzzy_variable('alteration', [0, 1], 'antecedent')
        alteration['sericite'] = fuzz.trimf(alteration.universe, [0.6, 0.8, 1.0])
        alteration['argillic'] = fuzz.trimf(alteration.universe, [0.4, 0.6, 0.8])
        alteration['propylitic'] = fuzz.trimf(alteration.universe, [0.2, 0.4, 0.6])
        
        gold_potential = self.create_fuzzy_variable('gold_potential', [0, 1], 'consequent')
        
        # Gold-specific rules
        gold_rules = [
            ctrl.Rule(geology['greenstone'] & magnetic['low_anomaly'] & 
                     geochemistry['pathfinder_high'] & alteration['sericite'], 
                     gold_potential['very_high']),
            
            ctrl.Rule(geology['greenstone'] & geochemistry['pathfinder_high'], 
                     gold_potential['high']),
            
            ctrl.Rule(geology['granite'] & magnetic['moderate_anomaly'] & 
                     alteration['argillic'], gold_potential['moderate']),
            
            ctrl.Rule(geology['sedimentary'] | (geochemistry['pathfinder_low'] & 
                     magnetic['high_anomaly']), gold_potential['low']),
            
            ctrl.Rule(geology['sedimentary'] & geochemistry['pathfinder_low'], 
                     gold_potential['very_low'])
        ]
        
        self.fuzzy_rules = gold_rules
        self.control_system = ctrl.ControlSystem(gold_rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        return self.control_system
    
    def evaluate_fuzzy_system(self, input_data_dict):
        """Evaluate the fuzzy system for given inputs"""
        if self.simulation is None:
            raise ValueError("No fuzzy system created. Call create_*_system() first.")
        
        # Set inputs
        for var_name, value in input_data_dict.items():
            if var_name in [var.label for var in self.simulation.ctrl.antecedents]:
                self.simulation.input[var_name] = np.clip(value, 0, 1)
        
        # Compute result
        try:
            self.simulation.compute()
            # Get output variable name (assuming single output)
            output_var = list(self.simulation.ctrl.consequents)[0]
            return self.simulation.output[output_var.label]
        except Exception as e:
            print(f"Error in fuzzy evaluation: {str(e)}")
            return 0.5  # Return neutral value on error
    
    def process_raster_data(self, input_rasters, output_path, system_type='general'):
        """Process raster data through fuzzy system"""
        # Create appropriate fuzzy system
        if system_type == 'gold':
            self.create_gold_prospectivity_system()
        else:
            self.create_mineral_prospectivity_system()
        
        # Load input rasters
        raster_data = {}
        reference_profile = None
        
        for var_name, raster_path in input_rasters.items():
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                raster_data[var_name] = data
                
                if reference_profile is None:
                    reference_profile = src.profile
        
        # Get dimensions
        shape = list(raster_data.values())[0].shape
        result = np.zeros(shape, dtype=np.float32)
        
        # Process each pixel
        for i in range(shape[0]):
            for j in range(shape[1]):
                inputs = {}
                valid_pixel = True
                
                for var_name, data in raster_data.items():
                    value = data[i, j]
                    if np.isfinite(value) and value >= 0:
                        inputs[var_name] = value
                    else:
                        valid_pixel = False
                        break
                
                if valid_pixel and inputs:
                    fuzzy_result = self.evaluate_fuzzy_system(inputs)
                    result[i, j] = fuzzy_result
                else:
                    result[i, j] = np.nan
        
        # Save result
        output_profile = reference_profile.copy()
        output_profile.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': np.nan
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(result, 1)
        
        return result
    
    def uncertainty_analysis(self, input_data, uncertainty_levels):
        """Perform uncertainty analysis using fuzzy sets"""
        results = {}
        
        for var_name, base_value in input_data.items():
            uncertainty = uncertainty_levels.get(var_name, 0.1)
            
            # Create uncertainty range
            lower_bound = max(0, base_value - uncertainty)
            upper_bound = min(1, base_value + uncertainty)
            
            # Sample values within uncertainty range
            sample_values = np.linspace(lower_bound, upper_bound, 11)
            fuzzy_outputs = []
            
            for sample_val in sample_values:
                test_inputs = input_data.copy()
                test_inputs[var_name] = sample_val
                output = self.evaluate_fuzzy_system(test_inputs)
                fuzzy_outputs.append(output)
            
            results[var_name] = {
                'base_output': self.evaluate_fuzzy_system(input_data),
                'uncertainty_range': [min(fuzzy_outputs), max(fuzzy_outputs)],
                'mean_output': np.mean(fuzzy_outputs),
                'std_output': np.std(fuzzy_outputs)
            }
        
        return results
    
    def sensitivity_analysis(self, base_inputs, perturbation=0.1):
        """Perform sensitivity analysis of fuzzy system"""
        base_output = self.evaluate_fuzzy_system(base_inputs)
        sensitivities = {}
        
        for var_name, base_value in base_inputs.items():
            # Positive perturbation
            perturbed_inputs = base_inputs.copy()
            perturbed_inputs[var_name] = min(1, base_value + perturbation)
            output_pos = self.evaluate_fuzzy_system(perturbed_inputs)
            
            # Negative perturbation
            perturbed_inputs[var_name] = max(0, base_value - perturbation)
            output_neg = self.evaluate_fuzzy_system(perturbed_inputs)
            
            # Calculate sensitivity
            sensitivity = (output_pos - output_neg) / (2 * perturbation)
            sensitivities[var_name] = {
                'sensitivity': sensitivity,
                'output_change_positive': output_pos - base_output,
                'output_change_negative': output_neg - base_output
            }
        
        return sensitivities
    
    def optimize_membership_functions(self, training_data, target_outputs):
        """Optimize membership function parameters using training data"""
        # This is a simplified optimization approach
        # In practice, you might use genetic algorithms or other optimization methods
        
        best_error = float('inf')
        best_params = None
        
        # Simple grid search for triangular membership function optimization
        param_ranges = np.linspace(0, 1, 21)
        
        for var_name, var in self.fuzzy_variables.items():
            if var_name in training_data:
                current_mf = list(var.terms.keys())[0]  # Optimize first membership function
                
                for a in param_ranges[::5]:  # Coarse grid
                    for b in param_ranges[::5]:
                        for c in param_ranges[::5]:
                            if a <= b <= c:
                                # Update membership function
                                var[current_mf] = fuzz.trimf(var.universe, [a, b, c])
                                
                                # Evaluate system performance
                                total_error = 0
                                for i, inputs in enumerate(training_data):
                                    predicted = self.evaluate_fuzzy_system(inputs)
                                    actual = target_outputs[i]
                                    total_error += (predicted - actual) ** 2
                                
                                if total_error < best_error:
                                    best_error = total_error
                                    best_params = {var_name: [a, b, c]}
        
        return best_params, best_error
    
    def export_fuzzy_system(self, output_path):
        """Export fuzzy system configuration"""
        system_config = {
            'variables': {},
            'rules': [],
            'membership_functions': {}
        }
        
        # Export variables and membership functions
        for var_name, var in self.fuzzy_variables.items():
            system_config['variables'][var_name] = {
                'universe': [float(var.universe.min()), float(var.universe.max())],
                'type': 'antecedent' if hasattr(var, 'terms') else 'consequent'
            }
            
            system_config['membership_functions'][var_name] = {}
            for mf_name, mf in var.terms.items():
                # Extract membership function parameters (simplified)
                system_config['membership_functions'][var_name][mf_name] = {
                    'type': 'triangular',  # Assuming triangular for simplicity
                    'parameters': mf.mf.tolist()
                }
        
        # Export rules (simplified representation)
        for i, rule in enumerate(self.fuzzy_rules):
            system_config['rules'].append({
                'rule_id': i,
                'description': str(rule),
                'label': rule.label if hasattr(rule, 'label') else None
            })
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(system_config, f, indent=2)
        
        return system_config
    
    def load_fuzzy_system(self, config_path):
        """Load fuzzy system from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recreate variables
        for var_name, var_config in config['variables'].items():
            universe_range = var_config['universe']
            var_type = var_config['type']
            self.create_fuzzy_variable(var_name, universe_range, var_type)
        
        # Recreate membership functions
        for var_name, mf_dict in config['membership_functions'].items():
            if var_name in self.fuzzy_variables:
                for mf_name, mf_config in mf_dict.items():
                    mf_type = mf_config['type']
                    parameters = mf_config['parameters']
                    self.create_custom_membership_function(var_name, mf_name, mf_type, parameters)
        
        return True

class FuzzySetOperations:
    """Fuzzy set operations and utilities"""
    
    @staticmethod
    def fuzzy_and(a, b):
        """Fuzzy AND operation (minimum)"""
        return np.minimum(a, b)
    
    @staticmethod
    def fuzzy_or(a, b):
        """Fuzzy OR operation (maximum)"""
        return np.maximum(a, b)
    
    @staticmethod
    def fuzzy_not(a):
        """Fuzzy NOT operation (complement)"""
        return 1 - a
    
    @staticmethod
    def fuzzy_alpha_cut(membership_values, alpha):
        """Apply alpha-cut to fuzzy set"""
        return (membership_values >= alpha).astype(float)
    
    @staticmethod
    def defuzzify_centroid(membership_function, universe):
        """Defuzzify using centroid method"""
        numerator = np.sum(membership_function * universe)
        denominator = np.sum(membership_function)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    @staticmethod
    def defuzzify_maximum(membership_function, universe):
        """Defuzzify using maximum method"""
        max_indices = np.where(membership_function == np.max(membership_function))[0]
        return universe[max_indices[0]]
    
    @staticmethod
    def fuzzy_similarity(set_a, set_b):
        """Calculate similarity between two fuzzy sets"""
        intersection = np.minimum(set_a, set_b)
        union = np.maximum(set_a, set_b)
        
        return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
