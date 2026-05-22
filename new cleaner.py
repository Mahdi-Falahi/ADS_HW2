import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# 0. LOAD AND INITIAL FILTERING
# ----------------------------------------------------------------------------
pd.set_option('display.max_columns', None)
df = pd.read_csv('./amesHousing.csv')
if 'Order' in df.columns:
    df.drop(columns='Order', inplace=True)

# Keep only residential zones
df = df[df['MS Zoning'].isin(['RL', 'RM', 'RH'])]

# Save a pristine copy of the raw data to use for "Before" visualizations later
old_df = df.copy()

print(f"Initial missing values: {df.isnull().sum().sum()}")

# ----------------------------------------------------------------------------
# 1. HANDLING MISSING VALUES (IMPUTATION)
# ----------------------------------------------------------------------------
# Categorical columns to "None"
cat_cols_to_none = [
    'Pool QC', 'Alley', 'Fence', 'Mas Vnr Type', 
    'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual', 
    'Garage Cond', 'Bsmt Exposure', 'BsmtFin Type 2', 'Bsmt Cond', 
    'Bsmt Qual', 'BsmtFin Type 1'
]
df[cat_cols_to_none] = df[cat_cols_to_none].fillna('None')

# Numerical columns to 0
num_cols_to_zero = [
    'Mas Vnr Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'BsmtFin SF 1', 
    'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Garage Cars', 'Garage Area'
]
df[num_cols_to_zero] = df[num_cols_to_zero].fillna(0)

# Lot Frontage: Impute by Neighborhood median
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))
df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())

# Electrical: Impute with Mode
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Garage Yr Blt: Replace with Year Built
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(df['Year Built'])

# Misc Feature extraction
df['Shed'] = (df['Misc Feature'] == 'Shed').astype(int)
df['Second Garage'] = (df['Misc Feature'] == 'Gar2').astype(int)
df.drop(columns='Misc Feature', inplace=True)

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (PART 1: RAW CORES & RATIOS)
# ----------------------------------------------------------------------------
# Perform size and space additions while values are still in raw square footage / counts
df['Total_SF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
df['Total_Bathrooms'] = df['Full Bath'] + 0.5 * df['Half Bath'] + df['Bsmt Full Bath'] + 0.5 * df['Bsmt Half Bath']
df['Total_Porch_SF'] = df['Open Porch SF'] + df['Enclosed Porch'] + df['3Ssn Porch'] + df['Screen Porch']

# Age calculations
df['House_Age'] = df['Yr Sold'] - df['Year Built']
df['Remodel_Age'] = df['Yr Sold'] - df['Year Remod/Add']
df['Is_Remodeled'] = (df['Year Built'] != df['Year Remod/Add']).astype(int)

# Structural ratios computed on clean, physical dimensions
df['Bsmt_Finished_Ratio'] = df['BsmtFin SF 1'] / (df['Total Bsmt SF'] + 1)
df['Rooms_Per_SF'] = df['TotRms AbvGrd'] / (df['Gr Liv Area'] + 1)
df['Bedroom_Ratio'] = df['Bedroom AbvGr'] / (df['TotRms AbvGrd'] + 1)

# Garage characteristics
df['Garage_Age'] = (df['Yr Sold'] - df['Garage Yr Blt']).clip(lower=0)
df['Garage_Space_Per_Car'] = df['Garage Area'] / (df['Garage Cars'] + 1)

# Property indicators (Flags)
df['Has_Garage'] = (df['Garage Cars'] > 0).astype(int)
df['Has_Basement'] = (df['Total Bsmt SF'] > 0).astype(int)
df['Has_Fireplace'] = (df['Fireplaces'] > 0).astype(int)
df['Has_Porch'] = (df['Total_Porch_SF'] > 0).astype(int)
df['Lot_Utilization'] = df['Gr Liv Area'] / (df['Lot Area'] + 1)

# ----------------------------------------------------------------------------
# 3. OUTLIER DETECTION & TREATMENT (TRANSFORMATIONS)
# ----------------------------------------------------------------------------
treatment_config = {
    # Winsorize heavy right-tail continuous columns
    "BsmtFin SF 2": ("winsorize", {"upper_q": 0.99}),
    "Mas Vnr Area": ("winsorize", {"upper_q": 0.99}),
    "Wood Deck SF": ("winsorize", {"upper_q": 0.99}),
    "Misc Val": ("winsorize", {"upper_q": 0.99}),
    "Total_Porch_SF": ("winsorize", {"upper_q": 0.99}),

    # IQR Capping for stable variance boundaries
    "Lot Frontage": ("iqr_cap", {}),
    "Bsmt Unf SF": ("iqr_cap", {}),
    "TotRms AbvGrd": ("iqr_cap", {}),
    "Bedroom AbvGr": ("iqr_cap", {}),
    "Garage_Space_Per_Car": ("iqr_cap", {}),
    "Rooms_Per_SF": ("iqr_cap", {}),
    "Bedroom_Ratio": ("iqr_cap", {}),
    "Bsmt_Finished_Ratio": ("iqr_cap", {}),
    "Lot_Utilization": ("iqr_cap", {}),

    # Log transformations to correct highly skewed monetary/spatial features
    "Gr Liv Area": ("log", {}),
    "Lot Area": ("log", {}),
    "Total_SF": ("log", {}),
    "SalePrice": ("log", {}),

    # Logical clipping for discrete variables
    "Kitchen AbvGr": ("clip", {"min": 0, "max": 3}),
    "Bsmt Half Bath": ("clip", {"min": 0, "max": 2}),
}

class OutlierTransformer:
    def __init__(self, config):
        self.config = config

    def iqr_bounds(self, series, k=1.5):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    def transform(self, data):
        data = data.copy()
        for col, (method, params) in self.config.items():
            if col not in data.columns:
                continue
            s = data[col]
            if method == "winsorize":
                upper = s.quantile(params.get("upper_q", 0.99))
                lower = s.quantile(params.get("lower_q", 0.00))
                data[col] = s.clip(lower, upper)
            elif method == "iqr_cap":
                lower, upper = self.iqr_bounds(s)
                data[col] = s.clip(lower, upper)
            elif method == "log":
                data[col] = np.log1p(s)
            elif method == "clip":
                data[col] = s.clip(params["min"], params["max"])
        return data

outlier_pipe = OutlierTransformer(treatment_config)
df = outlier_pipe.transform(df)

# ----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (PART 2: INTERACTION & NONLINEAR TERMS)
# ----------------------------------------------------------------------------
# Safe to execute now: Quality ranks remain strictly positive unscaled integers (1-10)
df['Qual_LivArea'] = df['Overall Qual'] * df['Gr Liv Area']
df['Qual_TotalSF'] = df['Overall Qual'] * df['Total_SF']
df['OverallQual2'] = df['Overall Qual'] ** 2
df['GrLivArea2'] = df['Gr Liv Area'] ** 2

# Drop original redundant parts to prevent multi-collinearity
redundant_features = [
    'Total Bsmt SF', '1st Flr SF', '2nd Flr SF',
    'Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath',
    'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
    'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'BsmtFin SF 1'
]
df.drop(columns=redundant_features, errors='ignore', inplace=True)

# ----------------------------------------------------------------------------
# 5. CATEGORICAL ENCODING
# ----------------------------------------------------------------------------
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmt_exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmt_fin_type_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
garage_finish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
functional_map = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
lot_shape_map = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
land_slope_map = {'Sev': 0, 'Mod': 1, 'Gtl': 2}

ordinal_features = {
    'Exter Qual': qual_map, 'Exter Cond': qual_map, 'Bsmt Qual': qual_map, 'Bsmt Cond': qual_map,
    'Heating QC': qual_map, 'Kitchen Qual': qual_map, 'Fireplace Qu': qual_map, 'Garage Qual': qual_map,
    'Garage Cond': qual_map, 'Pool QC': qual_map, 'Bsmt Exposure': bsmt_exposure_map,
    'BsmtFin Type 1': bsmt_fin_type_map, 'BsmtFin Type 2': bsmt_fin_type_map, 'Garage Finish': garage_finish_map,
    'Functional': functional_map, 'Street': {'Grvl': 0, 'Pave': 1}, 'Alley': {'None': 0, 'Grvl': 1, 'Pave': 2},
    'Paved Drive': {'N': 0, 'P': 1, 'Y': 2}, 'Lot Shape': lot_shape_map, 'Land Slope': land_slope_map,
    'Central Air': {'N': 0, 'Y': 1}
}

for col, mapping in ordinal_features.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

nominal_columns = [
    'MS Zoning', 'Land Contour', 'Utilities', 'Lot Config', 'Neighborhood', 'Condition 1', 
    'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 
    'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Electrical', 'Garage Type', 
    'Fence', 'Sale Type', 'Sale Condition'
]
df = pd.get_dummies(df, columns=nominal_columns, drop_first=True, dtype=int)

# ----------------------------------------------------------------------------
# 6. FEATURE SCALING (DELAYED TO THE END)
# ----------------------------------------------------------------------------
y = df['SalePrice']
X = df.drop('SalePrice', axis=1)

# Drop any leak/ID columns
X = X.drop(columns=[c for c in ['Unnamed: 0', 'PID'] if c in X.columns])

# Standard choices to exclude from scaling to preserve structural zero baselines
log_transformed = ['Gr Liv Area', 'Lot Area', 'Total_SF']
ordinal_cols = list(ordinal_features.keys()) + ['Overall Qual', 'Overall Cond']
binary_cols = [c for c in X.columns if X[c].nunique() == 2]

# Scale all remaining continuous and engineered continuous elements
cols_to_scale = [
    c for c in X.columns 
    if c not in log_transformed and c not in ordinal_cols and c not in binary_cols
]

scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Reassemble complete model dataset
df = pd.concat([X, y], axis=1)

# ----------------------------------------------------------------------------
# 7. FEATURE SELECTION: MUTUAL INFORMATION (MI)
# ----------------------------------------------------------------------------
print("="*70)
print("FEATURE SELECTION USING MUTUAL INFORMATION")
print("="*70)

X_features = df.drop('SalePrice', axis=1)
y_target = df['SalePrice']

mi_scores = mutual_info_regression(X_features, y_target, random_state=42)
mi_df = pd.DataFrame({'Feature': X_features.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)

print("\nTop 20 Features by Mutual Information Score:")
print(mi_df.head(20))

# Filter features
mi_threshold = 0.05
selected_features = mi_df[mi_df['MI_Score'] > mi_threshold]['Feature'].tolist()

X_selected = X_features[selected_features].copy()
X_selected['SalePrice'] = y_target

X_selected.to_csv('AmesHousing_MI_applied.csv', index=False)
print(f"\n✓ Selected {X_selected.shape[1] - 1} features. Saved clean data to 'AmesHousing_MI_applied.csv'.")