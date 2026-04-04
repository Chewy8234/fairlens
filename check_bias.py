import pandas as pd
import fairlens as fl

# Load one of the internal datasets (or use your own CSV)
# Let's assume there is a 'compas.csv' or similar in the datasets folder
df = pd.read_csv("datasets/compas.csv") 

# This is the "magic" command that identifies sensitive columns (Race, Gender, etc.)
fairobj = fl.FairnessProfiler(df, target_attribute="two_year_recid")

# Print the report to the console
fairobj.report()