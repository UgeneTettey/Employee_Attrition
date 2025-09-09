import pandas as pd
import numpy as np
import re
import unicodedata

class DataCleaner:
    """
    Comprehensive data cleaning pipeline for pandas DataFrames using OOP approach
    
    Features:
    - Duplicate removal
    - Text standardization
    - Missing value handling
    - Outlier management
    - Data type optimization
    - Detailed cleaning reports

    
    Usage examples:
    >>> cleaner = DataCleaner(df)
    >>> cleaned_df = cleaner.clean()
    >>> report = cleaner.report()
    >>> cleaner = DataCleaner(
        df,
        drop_duplicates=True,
        drop_columns=['ID', 'Timestamp'],
        handle_missing='auto',
        text_cleanup=True,
        numeric_outliers=False,
        verbose=True)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_missing: str = 'auto',
        text_cleanup: bool | list = True,
        numeric_outliers: bool = True,
        outlier_method: str = 'cap',
        iqr_multiplier: float = 1.5,
        category_threshold: int = 10,
        verbose: bool = True,
        missing_categorical_value: str = 'MISSING',
        drop_columns: list = None # columns to drop
    ):
        """
        Initialize data cleaner with configuration options
        
        Args:
            df: Input DataFrame to clean

            drop_duplicates: Remove duplicate rows (default: True)
            use case: drop_duplicates=True  :- will remove exact duplicate rows
            
            handle_missing: Strategy for missing values: 
                'auto' (median/mode), 'drop', 'fill' (mean/string), 'skip'
            1. 'auto' - Automatically fills numeric with median, categorical with mode
            2. 'drop' - Drops rows with any missing values 
            3. 'fill' - Fills numeric with mean, categorical with a specified value
            4. 'skip' - Skips missing value handling, useful for exploratory analysis
            use case: handle_missing='auto' 

                
            text_cleanup: Text cleaning: True (defaults), False, or list of operations:
                ['strip', 'lowercase', 'remove_special', 'normalize_unicode']

            numeric_outliers: Handle numeric outliers (default: True)

            outlier_method: 'cap' (winsorize) or 'remove' (delete rows)

            iqr_multiplier: IQR multiplier for outlier detection

            category_threshold: Unique value threshold for categorical conversion

            verbose: Print processing details

            missing_categorical_value: Fill value for categorical NaNs
        """
        # Validate parameters
        if handle_missing not in ['auto', 'drop', 'fill', 'skip']:
            raise ValueError("handle_missing must be 'auto', 'drop', 'fill', or 'skip'")
        if outlier_method not in ['cap', 'remove']:
            raise ValueError("outlier_method must be 'cap' or 'remove'")
        
        # Store configuration
        self.df = df.copy()
        self.original_df = df.copy()
        self.config = {
            'drop_duplicates': drop_duplicates,
            'handle_missing': handle_missing,
            'text_cleanup': text_cleanup,
            'numeric_outliers': numeric_outliers,
            'outlier_method': outlier_method,
            'iqr_multiplier': iqr_multiplier,
            'category_threshold': category_threshold,
            'verbose': verbose,
            'missing_categorical_value': missing_categorical_value,
            'drop_columns': drop_columns  #new config option
        }
        
        # Initialize reporting
        self.cleaning_report = {
            'initial_shape': self.df.shape,
            'duplicates_removed': 0,
            'missing_values_initial': self.df.isna().sum().sum(),
            'missing_values_final': 0,
            'type_conversions': {},
            'outliers': {},
            'rows_removed_outliers': 0,
            'columns_dropped':[], # new report field
            'final_shape': None
        }
    
    def log(self, message: str):
        """Conditional logging based on verbose setting"""
        if self.config['verbose']:
            print(message)
    
    # method for handling duplicates
    def _handle_duplicates(self):
        """Remove duplicate rows based on configuration"""
        if self.config['drop_duplicates']:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            removed = initial_count - len(self.df)
            self.cleaning_report['duplicates_removed'] = removed
            self.log(f"Removed {removed} duplicate rows")
    
    # method for cleaning text data
    def _clean_text_data(self):
        """Perform text cleaning operations"""
        if not self.config['text_cleanup']:
            return
            
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Set default operations if True is passed
        if self.config['text_cleanup'] is True:
            operations = ['strip', 'lowercase']
        else:
            operations = self.config['text_cleanup']
            
        for col in text_cols:
            try:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                
                # Apply selected text operations
                if 'strip' in operations:
                    self.df[col] = self.df[col].str.strip()
                if 'lowercase' in operations:
                    self.df[col] = self.df[col].str.lower()
                if 'remove_special' in operations:
                    self.df[col] = self.df[col].apply(
                        lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x)
                    )
                if 'normalize_unicode' in operations:
                    self.df[col] = self.df[col].apply(
                        lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')
                    )
                    
                self.log(f"Cleaned text in column: {col}")
            except Exception as e:
                self.log(f"Text cleaning failed for {col}: {str(e)}")
    
    # method for handling missing values
    def _handle_missing_values(self):
        """Address missing values based on selected strategy"""
        if self.config['handle_missing'] == 'skip' or self.cleaning_report['missing_values_initial'] == 0:
            return
            
        if self.config['verbose']:
            self.log("\nMissing values before handling:")
            self.log(self.df.isna().sum().to_string())
        
        if self.config['handle_missing'] == 'auto':
            for col in self.df.columns:
                if self.df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        fill_value = self.df[col].median()
                    else:
                        if not self.df[col].mode().empty:
                            fill_value = self.df[col].mode()[0]
                        else:
                            fill_value = self.config['missing_categorical_value']
                        if pd.isna(fill_value):
                            fill_value = self.config['missing_categorical_value']
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.log(f"Filled missing values in {col} with {fill_value}")
                    
        elif self.config['handle_missing'] == 'drop':
            initial = len(self.df)
            self.df = self.df.dropna().reset_index(drop=True)
            removed = initial - len(self.df)
            self.log(f"Removed {removed} rows with missing values")
            self.cleaning_report['missing_rows_removed'] = removed
            
        elif self.config['handle_missing'] == 'fill':
            for col in self.df.columns:
                if self.df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        fill_value = self.df[col].mean()
                    else:
                        fill_value = self.config['missing_categorical_value']
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.log(f"Filled missing values in {col} with {fill_value}")
        
        if self.config['verbose']:
            self.log("\nMissing values after handling:")
            self.log(self.df.isna().sum().to_string())
    
    # methods for optimizing data types
    def _optimize_data_types(self):
        """Convert to optimal data types and downcast where possible"""
        type_conversions = {}
        
        for col in self.df.columns:
            # Skip already categorical columns
            if pd.api.types.is_categorical_dtype(self.df[col]):
                continue
                
            # Convert to categorical if low cardinality
            if self.df[col].nunique() <= self.config['category_threshold'] and not pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].astype('category')
                self.log(f"Converted {col} to category dtype")
                type_conversions[col] = 'category'
                
            # Downcast numeric types
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                orig_type = self.df[col].dtype
                # Handle integers
                if pd.api.types.is_integer_dtype(self.df[col]):
                    self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
                # Handle floats
                else:
                    self.df[col] = pd.to_numeric(self.df[col], downcast='float')
                new_type = self.df[col].dtype
                if orig_type != new_type:
                    self.log(f"Downcasted {col} from {orig_type} to {new_type}")
                    type_conversions[col] = str(new_type)
        
        self.cleaning_report['type_conversions'] = type_conversions
    
    # method for handling outliers
    def _handle_outliers(self):
        """Detect and manage numeric outliers"""
        if not self.config['numeric_outliers']:
            return
            
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        initial_row_count = len(self.df)
        outlier_report = {}
        
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (self.config['iqr_multiplier'] * iqr)
            upper_bound = q3 + (self.config['iqr_multiplier'] * iqr)
            
            outliers = self.df[col][(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                if self.config['outlier_method'] == 'cap':
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                    self.log(f"Capped {outlier_count} outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                    outlier_report[col] = {
                        'method': 'capped',
                        'count': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                else:  # remove
                    self.df = self.df[~self.df.index.isin(outliers.index)]
                    self.log(f"Removed {outlier_count} outliers from {col}")
                    outlier_report[col] = {
                        'method': 'removed',
                        'count': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        self.cleaning_report['outliers'] = outlier_report
        self.cleaning_report['rows_removed_outliers'] = initial_row_count - len(self.df)
    
    # method for dropping specified columns
    def _drop_columns(self):
        """Drop specified columns from the DataFrame"""
        if self.config['drop_columns']:
            existing_cols = [col for col in self.config['drop_columns'] if col in self.df.columns]
            self.df = self.df.drop(columns=existing_cols)
            self.cleaning_report['columns_dropped'] = existing_cols
            self.log(f"Dropped columns: {existing_cols}")

    def _generate_final_report(self):
        """Complete reporting metrics and output summary"""
        self.cleaning_report['missing_values_final'] = self.df.isna().sum().sum()
        self.cleaning_report['final_shape'] = self.df.shape
        
        if self.config['verbose']:
            self.log("\n=== DATA CLEANING SUMMARY ===")
            self.log(f"Initial dimensions: {self.cleaning_report['initial_shape']}")
            self.log(f"Final dimensions: {self.cleaning_report['final_shape']}")
            self.log(f"Missing values handled: {self.cleaning_report['missing_values_initial']} → {self.cleaning_report['missing_values_final']}")
            
            if self.cleaning_report['duplicates_removed']:
                self.log(f"Duplicates removed: {self.cleaning_report['duplicates_removed']}")
            
            if 'outliers' in self.cleaning_report and self.cleaning_report['outliers']:
                self.log("\nOutlier handling:")
                for col, details in self.cleaning_report['outliers'].items():
                    self.log(f"- {col}: {details['count']} {details['method']} "
                          f"(bounds: [{details['lower_bound']:.2f}, {details['upper_bound']:.2f}])")
            
            if self.cleaning_report['type_conversions']:
                self.log("\nData type conversions:")
                for col, new_type in self.cleaning_report['type_conversions'].items():
                    self.log(f"- {col}: {new_type}")
    
    def clean(self) -> pd.DataFrame:
        """Execute the full cleaning pipeline"""
        self.log("Starting data cleaning process...")
        
        # Execute cleaning steps in sequence
        self._drop_columns()
        self._handle_duplicates()
        self._clean_text_data()
        self._handle_missing_values()
        self._optimize_data_types()
        self._handle_outliers()
        
        # Finalize reporting
        self._generate_final_report()
        self.log("\nData cleaning completed successfully!")
        
        return self.df
    
    def report(self) -> dict:
        """Get detailed cleaning report"""
        return self.cleaning_report
    
    def get_config(self) -> dict:
        """Get current configuration settings"""
        return self.config.copy()
    
    def reset(self):
        """Reset to original data while keeping configuration"""
        self.df = self.original_df.copy()
        self.cleaning_report = {
            'initial_shape': self.df.shape,
            'duplicates_removed': 0,
            'missing_values_initial': self.df.isna().sum().sum(),
            'missing_values_final': 0,
            'type_conversions': {},
            'outliers': {},
            'rows_removed_outliers': 0,
            'final_shape': None
        }

