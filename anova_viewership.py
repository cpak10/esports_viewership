import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

class ViewershipANOVA:
    """
    Set up the data to run ANOVA on viewership data
    """
    def clean_import(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the import csv for the stats needed for ANOVA

        Parameters
        ----------
        df : dataframe
            Dataframe to clean

        Outputs
        -------
        df_cleaned : dataframe
            Dataframe cleaned
        """
        df = df[["Match", "Peak Viewers"]]
        df = df[df["Match"].str.contains("vs")]
        df[["Team A", "Team B"]] = df["Match"].str.split(" vs ", expand = True)
        df_a = df[["Team A", "Peak Viewers"]].rename(columns={"Team A": "Team", "Peak Viewers": "Peak_Viewers"})
        df_b = df[["Team B", "Peak Viewers"]].rename(columns={"Team B": "Team", "Peak Viewers": "Peak_Viewers"})
        df = pd.concat([df_a, df_b], ignore_index = True, axis = 0)
        df["Intercept"] = 1

        df_cleaned = df
        return df_cleaned
    
 
    def run_anova(self, df: pd.DataFrame):
        """
        Run the ANOVA 

        Parameters
        ----------
        df : dataframe
            Dataframe to run ANOVA
        
        Outputs
        -------
        None
        """
        formula = "Peak_Viewers ~ C(Team)"
        model = ols(formula, df).fit()
        anova_table = sm.stats.anova_lm(model, typ = 1)
        print("\nANOVA table:\n", anova_table)
        posthoc = sm.stats.multicomp.pairwise_tukeyhsd(df["Peak_Viewers"], df["Team"])
        print("\nTukey HSD:\n", posthoc)



if __name__ == "__main__":
    calc = ViewershipANOVA()
    df_lec_2021_summer_import = pd.read_csv("lec_2021_summer.csv")
    df_lec_2021_summer = calc.clean_import(df_lec_2021_summer_import)
    calc.run_anova(df_lec_2021_summer)
    