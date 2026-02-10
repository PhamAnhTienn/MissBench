import sys
import argparse
import importlib

class ConfigManager:
    """ Manager for model configurations"""
    
    def __init__(self):
        self.available_models = ['gcnet', 'naive', 'redcore']
    
    def get_available_models(self):
        """Return list of available model names"""
        return self.available_models
        
    def import_model_config(self, model_name):
        """Dynamically import and return the config class for the given model name"""
        try:
            model_config_lib = importlib.import_module(f'config.{model_name}_config')
            return model_config_lib.Config
        except ImportError:
            print(f"Model config module 'config.{model_name}_config' not found!")
            print(f"Available models: {', '.join(self.get_available_models())}")
            sys.exit(1)
        
    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--base_model', type=str, default=None, help='chooses which model to use.')
        args, _ = parser.parse_known_args()

        Config = self.import_model_config(args.base_model)
        model_config = Config()
        model_args = model_config.get_args()
        model_args.base_model = args.base_model
    
        print(model_args.base_model)
        
        return model_args
    
    
    
if __name__ == "__main__":
    config_manager = ConfigManager()
    # config = config_manager.get_config()
    args = config_manager.get_args()
    print(args)