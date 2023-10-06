import configparser
import argparse
from train import train_model

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNABERT Benchmark experiments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Configuration file location")
    args = parser.parse_args()
    config_object = configparser.ConfigParser() 
    # Read config.ini file
    config_object.read(args.config)
    # Convert to dictionary
    config_dict = {}
    for section in config_object.sections():
        config_dict[section] = {}
        for option in config_object.options(section):
            config_dict[section][option] = config_object.get(section, option)
    train_model(config_dict=config_dict)
    
# TODO
# **
# TODO 1: Pretty print
# TODO 2: Single file results
# TODO 3: Metrics function
# TODO 4: Figures creation and save
# TODO 5: Modify reading of config file to include default fields
# TODO 6: Add config file verification option
# TODO 7: Add DNABERT path and sequence length as parameter
# TODO 8: Add myprint function
# TODO 9: Add custom head classifier to Model B option
# **
    
