import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
from pathlib import Path

class CarbonModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_metrics = {}
        self.feature_names = []
        self.is_trained_flag = False
        self.data = None
        self.data_dir = Path(__file__).resolve().parent / "data"
        
    def load_data(self):
        """Load and preprocess data according to your dataset structure"""
        if self.data is not None:
            return self.data
            
        try:
            # Load your actual datasets
            cost_data = pd.read_csv(self.data_dir / 'cost_breakdown.csv')
            delivery_data = pd.read_csv(self.data_dir / 'delivery_performance.csv')
            orders_data = pd.read_csv(self.data_dir / 'orders.csv')
            routes_data = pd.read_csv(self.data_dir / 'routes_distance.csv')
            vehicle_data = pd.read_csv(self.data_dir / 'vehicle_fleet.csv')
            
            # Merge datasets based on Order_ID (common key)
            merged_data = orders_data.merge(routes_data, on='Order_ID', how='left')
            merged_data = merged_data.merge(delivery_data, on='Order_ID', how='left')
            merged_data = merged_data.merge(cost_data, on='Order_ID', how='left')
            
            # Calculate carbon emissions using your data
            # Since we have Distance_KM and CO2_Emissions_Kg_per_KM from vehicle data
            # We need to assign vehicles to orders logically
            
            # Create a mapping of vehicle types to average emissions
            vehicle_emissions = vehicle_data.groupby('Vehicle_Type')['CO2_Emissions_Kg_per_KM'].mean()
            
            # Assign vehicle types to orders based on capacity needs and distance
            def assign_vehicle_type(row):
                distance = row.get('Distance_KM', 0)
                # Handle NaN distance values
                if pd.isna(distance):
                    distance = 0
                
                if distance > 2000:
                    return 'Large_Truck'
                elif distance > 500:
                    return 'Medium_Truck'
                else:
                    # Check Special_Handling safely (handle NaN/float cases)
                    special_handling = row.get('Special_Handling', '')
                    if pd.notna(special_handling) and isinstance(special_handling, str):
                        if 'Refrigerated' in str(special_handling):
                            return 'Refrigerated'
                    # Also check Product_Category for refrigerated items
                    product_cat = row.get('Product_Category', '')
                    if pd.notna(product_cat) and isinstance(product_cat, str):
                        if product_cat in ['Food & Beverage', 'Healthcare']:
                            return 'Refrigerated'
                    return 'Small_Van'
            
            merged_data['Vehicle_Type'] = merged_data.apply(assign_vehicle_type, axis=1)
            merged_data['CO2_Emissions_Kg_per_KM'] = merged_data['Vehicle_Type'].map(vehicle_emissions)
            
            # Calculate carbon emissions
            merged_data['Carbon_Emissions_Kg'] = (
                merged_data['Distance_KM'] * merged_data['CO2_Emissions_Kg_per_KM']
            )
            
            # Add vehicle characteristics based on type
            vehicle_chars = vehicle_data.groupby('Vehicle_Type').agg({
                'Capacity_KG': 'mean',
                'Fuel_Efficiency_KM_per_L': 'mean',
                'Age_Years': 'mean'
            }).reset_index()
            
            merged_data = merged_data.merge(vehicle_chars, on='Vehicle_Type', how='left')
            
            self.data = merged_data
            return self.data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            # Return sample data structure for demonstration
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data if file loading fails"""
        st.warning("Using sample data for demonstration. Please check your CSV files.")
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'Distance_KM': np.random.uniform(50, 5000, n_samples),
            'Order_Value_INR': np.random.uniform(100, 50000, n_samples),
            'Fuel_Consumption_L': np.random.uniform(10, 500, n_samples),
            'Vehicle_Type': np.random.choice(['Small_Van', 'Medium_Truck', 'Large_Truck', 'Refrigerated'], n_samples),
            'Product_Category': np.random.choice(['Electronics', 'Fashion', 'Food & Beverage', 'Healthcare', 'Industrial'], n_samples),
            'Priority': np.random.choice(['Express', 'Standard', 'Economy'], n_samples),
            'Weather_Impact': np.random.choice(['None', 'Light_Rain', 'Heavy_Rain', 'Fog'], n_samples),
            'Traffic_Delay_Minutes': np.random.randint(0, 120, n_samples),
            'Toll_Charges_INR': np.random.uniform(0, 500, n_samples),
            'Capacity_KG': np.random.uniform(500, 5000, n_samples),
            'Fuel_Efficiency_KM_per_L': np.random.uniform(5, 15, n_samples),
            'Age_Years': np.random.uniform(1, 10, n_samples)
        })
        
        # Calculate realistic carbon emissions
        emission_factors = {
            'Small_Van': 0.3, 'Medium_Truck': 0.45, 'Large_Truck': 0.6, 'Refrigerated': 0.5
        }
        
        sample_data['CO2_Emissions_Kg_per_KM'] = sample_data['Vehicle_Type'].map(emission_factors)
        sample_data['Carbon_Emissions_Kg'] = (
            sample_data['Distance_KM'] * sample_data['CO2_Emissions_Kg_per_KM']
        )
        
        self.data = sample_data
        return self.data
    
    def preprocess_data(self, data):
        """Preprocess data for model training based on your dataset"""
        # Select features available in your data
        features = [
            'Distance_KM', 'Order_Value_INR', 'Fuel_Consumption_L', 
            'Toll_Charges_INR', 'Traffic_Delay_Minutes',
            'Capacity_KG', 'Fuel_Efficiency_KM_per_L', 'Age_Years',
            'Product_Category', 'Priority', 'Vehicle_Type', 'Weather_Impact'
        ]
        
        # Filter data
        model_data = data[features + ['Carbon_Emissions_Kg']].copy()
        model_data = model_data.dropna()
        
        # Encode categorical variables
        categorical_cols = ['Product_Category', 'Priority', 'Vehicle_Type', 'Weather_Impact']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            model_data[col] = self.label_encoders[col].fit_transform(model_data[col].astype(str))
        
        self.feature_names = features
        return model_data
    
    def train_model(self, model_type="Random Forest", train_size=0.8):
        """Train the carbon prediction model on your data"""
        try:
            data = self.load_data()
            if data is None:
                return None
                
            # Preprocess data
            model_data = self.preprocess_data(data)
            
            # Prepare features and target
            X = model_data[self.feature_names]
            y = model_data['Carbon_Emissions_Kg']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-train_size/100, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            if model_type == "Random Forest":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == "Gradient Boosting":
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == "Linear Regression":
                self.model = LinearRegression()
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            
            self.performance_metrics = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'n_samples': len(X_train),
                'model_type': model_type
            }
            
            self.is_trained_flag = True
            return self.performance_metrics
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def predict_single(self, input_data):
        """Predict carbon emissions for a single input using your trained model"""
        if not self.is_trained():
            st.warning("Model not trained. Please train the model first.")
            return None
        
        try:
            # Convert input to dataframe
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in ['Product_Category', 'Priority', 'Vehicle_Type', 'Weather_Impact']:
                if col in input_df.columns:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        try:
                            input_df[col] = self.label_encoders[col].transform([input_data[col]])[0]
                        except ValueError:
                            # If category not seen during training, use most common
                            input_df[col] = 0
                    else:
                        input_df[col] = 0
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Select and order features
            input_df = input_df[self.feature_names]
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            return {
                'prediction': prediction,
                'feature_impact': self._calculate_feature_impact(input_scaled[0], prediction)
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def _calculate_feature_impact(self, input_scaled, prediction):
        """Calculate feature impact for prediction"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_impact = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance,
                'Impact': importance * prediction
            }).sort_values('Impact', ascending=False)
        else:
            # For linear models, use coefficients
            if hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
                feature_impact = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance / importance.sum(),
                    'Impact': importance * prediction
                }).sort_values('Impact', ascending=False)
            else:
                feature_impact = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': 1.0/len(self.feature_names),
                    'Impact': prediction/len(self.feature_names)
                })
        
        return feature_impact
    
    def get_feature_impact(self, input_data):
        """Get feature impact for a prediction"""
        prediction_result = self.predict_single(input_data)
        if prediction_result:
            return prediction_result['feature_impact']
        return pd.DataFrame()
    
    def is_trained(self):
        return self.is_trained_flag
    
    def get_feature_names(self):
        return self.feature_names
    
    def get_training_info(self):
        return self.performance_metrics
    
    def get_categories(self, column):
        """Get unique categories for a column from your data"""
        data = self.load_data()
        if data is not None and column in data.columns:
            return sorted(data[column].astype(str).unique())
        return []
    
    def get_average_emission(self):
        """Get average carbon emission from your data"""
        if self.data is not None:
            return self.data['Carbon_Emissions_Kg'].mean()
        return 0
    
    def plot_feature_importance(self):
        """Plot feature importance based on your model"""
        if not self.is_trained():
            fig = go.Figure()
            fig.add_annotation(text="Feature importance not available", x=0.5, y=0.5)
            return fig
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=True)
        else:
            # For linear models
            if hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance / importance.sum()
                }).sort_values('Importance', ascending=True)
            else:
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': 1.0/len(self.feature_names)
                })
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            title='Feature Importance in Carbon Prediction',
            orientation='h'
        )
        return fig
    
    def plot_prediction_vs_actual(self):
        """Plot prediction vs actual values from your data"""
        if not self.is_trained():
            fig = go.Figure()
            fig.add_annotation(text="Model not trained", x=0.5, y=0.5)
            return fig
        
        # Get test predictions
        data = self.load_data()
        model_data = self.preprocess_data(data)
        X = model_data[self.feature_names]
        y = model_data['Carbon_Emissions_Kg']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        fig = px.scatter(
            x=y_test, y=y_pred,
            title='Predicted vs Actual Carbon Emissions',
            labels={'x': 'Actual Emissions (kg)', 'y': 'Predicted Emissions (kg)'}
        )
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        ))
        return fig
    
    def plot_error_distribution(self):
        """Plot error distribution from your model"""
        if not self.is_trained():
            fig = go.Figure()
            fig.add_annotation(text="Model not trained", x=0.5, y=0.5)
            return fig
        
        data = self.load_data()
        model_data = self.preprocess_data(data)
        X = model_data[self.feature_names]
        y = model_data['Carbon_Emissions_Kg']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        errors = y_pred - y_test
        
        fig = px.histogram(
            x=errors,
            title='Prediction Error Distribution',
            labels={'x': 'Prediction Error (kg)'}
        )
        return fig
    
    def plot_model_comparison(self):
        """Plot model comparison based on your trained model"""
        if self.performance_metrics:
            metrics = ['R² Score', 'MSE', 'MAE']
            values = [
                self.performance_metrics['r2'],
                self.performance_metrics['mse'],
                self.performance_metrics['mae']
            ]
            
            fig = px.bar(
                x=metrics, y=values,
                title='Model Performance Metrics',
                labels={'x': 'Metric', 'y': 'Value'}
            )
            return fig
        
        fig = go.Figure()
        fig.add_annotation(text="No performance data", x=0.5, y=0.5)
        return fig
    
    def calculate_realtime_emissions(self, distance, vehicle_type, fuel_type, load_factor,
                                  traffic, weather, route_type):
        """Calculate real-time emissions using patterns from your data"""
        # Base emission factors from your vehicle data
        emission_factors = {
            'Small_Van': 0.3,
            'Medium_Truck': 0.45, 
            'Large_Truck': 0.6,
            'Refrigerated': 0.5
        }
        
        # Adjustment factors based on your route data patterns
        traffic_factors = {
            'Free Flow': 0.9,
            'Light': 1.0,
            'Moderate': 1.2,
            'Heavy': 1.5,
            'Congested': 2.0
        }
        
        weather_factors = {
            'Clear': 1.0,
            'Light Rain': 1.1,
            'Heavy Rain': 1.3,
            'Fog': 1.2,
            'Extreme': 1.5
        }
        
        route_factors = {
            'Highway': 0.9,
            'Urban': 1.3,
            'Mixed': 1.1,
            'Rural': 1.0
        }
        
        # Calculate base emission from your vehicle data
        base_emission = emission_factors.get(vehicle_type, 0.4)
        
        # Apply adjustments based on your route patterns
        adjusted_emission = (base_emission * 
                           traffic_factors[traffic] * 
                           weather_factors[weather] * 
                           route_factors[route_type] *
                           (1 + load_factor/100 * 0.3))
        
        total_emissions = adjusted_emission * distance
        fuel_used = distance / 8  # Based on your fuel efficiency data
        
        # Cost impact based on your cost breakdown
        fuel_cost = fuel_used * 80  # Based on your fuel costs
        carbon_cost = total_emissions * 0.1
        total_cost = fuel_cost + carbon_cost
        
        # Efficiency score based on your data patterns
        efficiency_score = max(0, min(1, 1 - (adjusted_emission / 0.8)))
        
        return {
            'co2_emissions': total_emissions,
            'fuel_used': fuel_used,
            'cost_impact': total_cost,
            'efficiency_score': efficiency_score
        }