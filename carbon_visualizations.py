import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

class CarbonVisualizations:
    def __init__(self, model):
        self.model = model
        self.data = model.load_data()
        self.data_dir = Path(__file__).resolve().parent / "data"
    
    def _prepare_plot_data(self, data, size_column=None, required_columns=None):
        """Helper method to prepare data for plotting by removing NaN values"""
        plot_data = data.copy()
        
        # Filter out NaN in size column if specified
        if size_column and size_column in plot_data.columns:
            plot_data = plot_data[plot_data[size_column].notna()]
        
        # Filter out NaN in required columns
        if required_columns:
            for col in required_columns:
                if col in plot_data.columns:
                    plot_data = plot_data[plot_data[col].notna()]
        
        return plot_data
    
    def get_carriers(self):
        """Get unique carriers from your delivery_performance data"""
        if 'Carrier' in self.data.columns:
            return sorted(self.data['Carrier'].dropna().unique())
        return []
    
    def get_vehicle_types(self):
        """Get unique vehicle types from your vehicle_fleet data"""
        if 'Vehicle_Type' in self.data.columns:
            return sorted(self.data['Vehicle_Type'].dropna().unique())
        return []
    
    def get_product_categories(self):
        """Get unique product categories from your orders data"""
        if 'Product_Category' in self.data.columns:
            return sorted(self.data['Product_Category'].dropna().unique())
        return []
    
    def get_carbon_metrics(self, carrier_filter, vehicle_filter, product_filter):
        """Calculate carbon metrics based on your actual data"""
        filtered_data = self._apply_filters(carrier_filter, vehicle_filter, product_filter)
        
        if filtered_data.empty:
            return {
                'total_co2': 0,
                'avg_per_order': 0,
                'intensity': 0,
                'efficiency_score': 0
            }
        
        total_co2 = filtered_data['Carbon_Emissions_Kg'].sum()
        avg_per_order = filtered_data['Carbon_Emissions_Kg'].mean()
        intensity = total_co2 / filtered_data['Order_Value_INR'].sum() if filtered_data['Order_Value_INR'].sum() > 0 else 0
        
        # Efficiency score based on your actual emissions data
        if 'Distance_KM' in filtered_data.columns and 'Carbon_Emissions_Kg' in filtered_data.columns:
            total_distance = filtered_data['Distance_KM'].sum()
            total_emissions = filtered_data['Carbon_Emissions_Kg'].sum()
            avg_emission_per_km = total_emissions / total_distance if total_distance > 0 else 0
            efficiency_score = max(0, 100 - (avg_emission_per_km * 100))
        else:
            efficiency_score = 0
        
        return {
            'total_co2': total_co2,
            'avg_per_order': avg_per_order,
            'intensity': intensity,
            'efficiency_score': efficiency_score
        }
    
    def _apply_filters(self, carrier, vehicle, product):
        """Apply filters to your data"""
        filtered_data = self.data.copy()
        if carrier != 'All' and 'Carrier' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Carrier'] == carrier]
        if vehicle != 'All' and 'Vehicle_Type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Vehicle_Type'] == vehicle]
        if product != 'All' and 'Product_Category' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Product_Category'] == product]
        return filtered_data
    
    def plot_emissions_by_vehicle(self):
        """Plot carbon emissions by vehicle type using your vehicle_fleet data"""
        if 'Vehicle_Type' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Vehicle emissions data not available")
        
        vehicle_emissions = self.data.groupby('Vehicle_Type')['Carbon_Emissions_Kg'].agg(['sum', 'mean', 'count']).reset_index()
        vehicle_emissions.columns = ['Vehicle_Type', 'Total_CO2', 'Avg_CO2', 'Order_Count']
        
        fig = px.bar(
            vehicle_emissions,
            x='Vehicle_Type',
            y='Avg_CO2',
            color='Total_CO2',
            title='Carbon Emissions by Vehicle Type',
            labels={'Avg_CO2': 'Average CO₂ Emissions (kg)', 'Vehicle_Type': 'Vehicle Type'}
        )
        return fig
    
    def plot_carrier_comparison(self):
        """Plot carbon emissions comparison by carrier using your delivery_performance data"""
        if 'Carrier' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Carrier data not available")
        
        carrier_emissions = self.data.groupby('Carrier').agg({
            'Carbon_Emissions_Kg': ['sum', 'mean'],
            'Delivery_Cost_INR': 'mean',
            'Customer_Rating': 'mean'
        }).reset_index()
        
        carrier_emissions.columns = ['Carrier', 'Total_CO2', 'Avg_CO2', 'Avg_Cost', 'Avg_Rating']
        
        # Clean data - remove NaN values
        carrier_emissions = carrier_emissions[
            carrier_emissions[['Total_CO2', 'Avg_CO2', 'Avg_Cost', 'Avg_Rating']].notna().all(axis=1)
        ]
        
        if carrier_emissions.empty:
            return self._create_empty_plot("No valid carrier data available")
        
        fig = px.scatter(
            carrier_emissions,
            x='Avg_CO2',
            y='Avg_Cost',
            size='Total_CO2',
            color='Avg_Rating',
            hover_name='Carrier',
            title='Carrier Performance: Cost vs Carbon Emissions',
            labels={'Avg_CO2': 'Average CO₂ per Order (kg)', 'Avg_Cost': 'Average Delivery Cost (INR)'}
        )
        return fig
    
    def plot_product_emissions(self):
        """Plot carbon emissions by product category using your orders data"""
        if 'Product_Category' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Product category data not available")
        
        product_emissions = self.data.groupby('Product_Category')['Carbon_Emissions_Kg'].sum().reset_index()
        
        fig = px.pie(
            product_emissions,
            values='Carbon_Emissions_Kg',
            names='Product_Category',
            title='Carbon Emissions Distribution by Product Category',
            hole=0.4
        )
        return fig
    
    def plot_priority_impact(self):
        """Plot impact of delivery priority on carbon emissions"""
        if 'Priority' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Priority data not available")
        
        priority_impact = self.data.groupby('Priority').agg({
                    'Carbon_Emissions_Kg': 'mean',
            'Actual_Delivery_Days': 'mean',
            'Fuel_Consumption_L': 'mean'
                }).reset_index()
        
        fig = px.bar(
            priority_impact,
            x='Priority',
            y='Carbon_Emissions_Kg',
            color='Actual_Delivery_Days',
            title='Impact of Delivery Priority on Carbon Emissions',
            labels={'Carbon_Emissions_Kg': 'Average CO₂ Emissions (kg)', 'Priority': 'Delivery Priority'}
        )
        return fig
    
    def plot_vehicle_efficiency(self):
        """Plot vehicle efficiency analysis using your vehicle_fleet data"""
        if 'Vehicle_Type' not in self.data.columns or 'CO2_Emissions_Kg_per_KM' not in self.data.columns:
            return self._create_empty_plot("Vehicle efficiency data not available")
        
        vehicle_efficiency = self.data.groupby('Vehicle_Type').agg({
            'CO2_Emissions_Kg_per_KM': 'mean',
            'Fuel_Efficiency_KM_per_L': 'mean',
            'Carbon_Emissions_Kg': 'mean'
        }).reset_index()
        
        # Clean data - remove NaN values
        vehicle_efficiency = vehicle_efficiency[
            vehicle_efficiency[['Carbon_Emissions_Kg', 'Fuel_Efficiency_KM_per_L', 'CO2_Emissions_Kg_per_KM']].notna().all(axis=1)
        ]
        
        if vehicle_efficiency.empty:
            return self._create_empty_plot("No valid vehicle efficiency data available")
        
        fig = px.scatter(
            vehicle_efficiency,
            x='Fuel_Efficiency_KM_per_L',
            y='CO2_Emissions_Kg_per_KM',
            size='Carbon_Emissions_Kg',
            color='Vehicle_Type',
            title='Vehicle Efficiency: Fuel vs Carbon Emissions',
            labels={
                'Fuel_Efficiency_KM_per_L': 'Fuel Efficiency (KM/L)',
                'CO2_Emissions_Kg_per_KM': 'CO₂ Emissions (kg/km)'
            }
        )
        return fig
    
    def plot_fuel_consumption_analysis(self):
        """Plot fuel consumption analysis using your routes data"""
        if 'Fuel_Consumption_L' not in self.data.columns or 'Distance_KM' not in self.data.columns:
            return self._create_empty_plot("Fuel consumption data not available")
        
        # Calculate fuel efficiency
        self.data['Actual_Fuel_Efficiency'] = self.data['Distance_KM'] / self.data['Fuel_Consumption_L']
        
        # Filter out NaN values for the size column and prepare data for plotting
        plot_data = self.data.copy()
        if 'Carbon_Emissions_Kg' in plot_data.columns:
            # Fill NaN with 0 or filter them out
            plot_data = plot_data[plot_data['Carbon_Emissions_Kg'].notna()]
            if plot_data.empty:
                return self._create_empty_plot("No valid data points for fuel consumption analysis")
        
        fig = px.scatter(
            plot_data,
            x='Distance_KM',
            y='Fuel_Consumption_L',
            color='Vehicle_Type',
            size='Carbon_Emissions_Kg',
            title='Fuel Consumption vs Distance',
            trendline='lowess',
            labels={
                'Distance_KM': 'Distance (KM)',
                'Fuel_Consumption_L': 'Fuel Consumption (L)'
            }
        )
        return fig
    
    def plot_capacity_analysis(self):
        """Plot vehicle capacity utilization analysis"""
        if 'Capacity_KG' not in self.data.columns or 'Order_Value_INR' not in self.data.columns:
            return self._create_empty_plot("Capacity data not available")
        
        # Create utilization metric (simplified)
        self.data['Utilization_Rate'] = (self.data['Order_Value_INR'] / self.data['Order_Value_INR'].max()) * 100
        
        # Clean data - remove NaN values in key columns
        plot_data = self.data[
            self.data[['Capacity_KG', 'Carbon_Emissions_Kg', 'Distance_KM']].notna().all(axis=1)
        ]
        
        if plot_data.empty:
            return self._create_empty_plot("No valid capacity data available")
        
        fig = px.scatter(
            plot_data,
            x='Capacity_KG',
            y='Carbon_Emissions_Kg',
            color='Utilization_Rate',
            size='Distance_KM',
            title='Vehicle Capacity vs Carbon Emissions',
            labels={
                'Capacity_KG': 'Vehicle Capacity (KG)',
                'Carbon_Emissions_Kg': 'CO₂ Emissions (kg)',
                'Utilization_Rate': 'Utilization Rate (%)'
            }
        )
        return fig
    
    def plot_emissions_by_location(self):
        """Plot carbon emissions by origin and destination using your orders data"""
        if 'Origin' not in self.data.columns or 'Destination' not in self.data.columns:
            return self._create_empty_plot("Location data not available")
        
        # Aggregate by origin
        origin_emissions = self.data.groupby('Origin')['Carbon_Emissions_Kg'].sum().reset_index()
        destination_emissions = self.data.groupby('Destination')['Carbon_Emissions_Kg'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Origin Emissions',
            x=origin_emissions['Origin'],
            y=origin_emissions['Carbon_Emissions_Kg'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Destination Emissions',
            x=destination_emissions['Destination'],
            y=destination_emissions['Carbon_Emissions_Kg'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Carbon Emissions by Location',
            xaxis_title='Location',
            yaxis_title='Total CO₂ Emissions (kg)',
            barmode='group'
        )
        
        return fig
    
    def plot_route_efficiency(self):
        """Plot route efficiency analysis using your routes data"""
        if 'Route' not in self.data.columns or 'Distance_KM' not in self.data.columns:
            return self._create_empty_plot("Route data not available")
        
        route_efficiency = self.data.groupby('Route').agg({
            'Carbon_Emissions_Kg': 'mean',
            'Distance_KM': 'mean',
            'Fuel_Consumption_L': 'mean',
            'Traffic_Delay_Minutes': 'mean'
        }).reset_index()
        
        route_efficiency['Efficiency_Ratio'] = route_efficiency['Carbon_Emissions_Kg'] / route_efficiency['Distance_KM']
        
        # Get top 15 routes for better visualization
        top_routes = route_efficiency.nlargest(15, 'Carbon_Emissions_Kg')
        
        fig = px.bar(
            top_routes,
            x='Route',
            y='Efficiency_Ratio',
            color='Traffic_Delay_Minutes',
            title='Route Efficiency Analysis (Top 15 Routes by Emissions)',
            labels={
                'Efficiency_Ratio': 'CO₂ per KM (kg/km)',
                'Route': 'Route',
                'Traffic_Delay_Minutes': 'Avg Traffic Delay (min)'
            }
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def plot_weather_impact(self):
        """Plot weather impact on carbon emissions using your routes data"""
        if 'Weather_Impact' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Weather impact data not available")
        
        weather_impact = self.data.groupby('Weather_Impact').agg({
            'Carbon_Emissions_Kg': 'mean',
            'Fuel_Consumption_L': 'mean',
            'Traffic_Delay_Minutes': 'mean'
        }).reset_index()
        
        fig = px.bar(
            weather_impact,
            x='Weather_Impact',
            y='Carbon_Emissions_Kg',
            color='Traffic_Delay_Minutes',
            title='Weather Impact on Carbon Emissions',
            labels={
                'Carbon_Emissions_Kg': 'Average CO₂ Emissions (kg)',
                'Weather_Impact': 'Weather Conditions'
            }
        )
        return fig
    
    def plot_traffic_impact(self):
        """Plot traffic impact on carbon emissions"""
        if 'Traffic_Delay_Minutes' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Traffic delay data not available")
        
        # Create bins for traffic delay
        self.data['Traffic_Delay_Bin'] = pd.cut(
            self.data['Traffic_Delay_Minutes'],
            bins=[0, 15, 30, 60, 120, float('inf')],
            labels=['0-15min', '15-30min', '30-60min', '60-120min', '120+min']
        )
        
        traffic_impact = self.data.groupby('Traffic_Delay_Bin')['Carbon_Emissions_Kg'].mean().reset_index()
        
        fig = px.bar(
            traffic_impact,
            x='Traffic_Delay_Bin',
            y='Carbon_Emissions_Kg',
            title='Traffic Delay Impact on Carbon Emissions',
            labels={
                'Carbon_Emissions_Kg': 'Average CO₂ Emissions (kg)',
                'Traffic_Delay_Bin': 'Traffic Delay Duration'
            }
        )
        return fig
    
    def plot_distance_vs_emissions(self):
        """Plot distance vs carbon emissions relationship"""
        if 'Distance_KM' not in self.data.columns or 'Carbon_Emissions_Kg' not in self.data.columns:
            return self._create_empty_plot("Distance data not available")
        
        # Clean data - remove NaN values in key columns
        plot_data = self.data[
            self.data[['Distance_KM', 'Carbon_Emissions_Kg', 'Fuel_Consumption_L']].notna().all(axis=1)
        ]
        
        if plot_data.empty:
            return self._create_empty_plot("No valid distance/emissions data available")
        
        fig = px.scatter(
            plot_data,
            x='Distance_KM',
            y='Carbon_Emissions_Kg',
            color='Vehicle_Type',
            size='Fuel_Consumption_L',
            title='Distance vs Carbon Emissions',
            trendline='lowess',
            labels={
                'Distance_KM': 'Distance (KM)',
                'Carbon_Emissions_Kg': 'CO₂ Emissions (kg)'
            }
        )
        return fig
    
    def get_high_impact_routes(self):
        """Identify high impact routes for optimization based on your data"""
        if 'Route' not in self.data.columns:
            return pd.DataFrame()
        
        route_analysis = self.data.groupby('Route').agg({
            'Carbon_Emissions_Kg': ['sum', 'mean'],
            'Distance_KM': 'mean',
            'Fuel_Consumption_L': 'mean',
            'Traffic_Delay_Minutes': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        
        route_analysis.columns = [
            'Route', 'Total_CO2', 'Avg_CO2', 'Avg_Distance', 
            'Avg_Fuel_Consumption', 'Avg_Traffic_Delay', 'Order_Count'
        ]
        
        route_analysis['Efficiency_Ratio'] = route_analysis['Avg_CO2'] / route_analysis['Avg_Distance']
        
        # Identify high impact routes (high emissions and high frequency)
        high_impact = route_analysis[
            (route_analysis['Total_CO2'] > route_analysis['Total_CO2'].quantile(0.7)) &
            (route_analysis['Order_Count'] > route_analysis['Order_Count'].quantile(0.7))
        ]
        
        return high_impact.sort_values('Total_CO2', ascending=False).head(10)
    
    def plot_delivery_performance(self):
        """Plot delivery performance metrics from your delivery_performance data"""
        if 'Delivery_Status' not in self.data.columns or 'Actual_Delivery_Days' not in self.data.columns:
            return self._create_empty_plot("Delivery performance data not available")
        
        performance_metrics = self.data.groupby('Delivery_Status').agg({
            'Carbon_Emissions_Kg': 'mean',
            'Actual_Delivery_Days': 'mean',
            'Customer_Rating': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        
        # Clean data - remove NaN values
        performance_metrics = performance_metrics[
            performance_metrics[['Carbon_Emissions_Kg', 'Actual_Delivery_Days', 'Customer_Rating', 'Order_ID']].notna().all(axis=1)
        ]
        
        if performance_metrics.empty:
            return self._create_empty_plot("No valid delivery performance data available")
        
        fig = px.scatter(
            performance_metrics,
            x='Actual_Delivery_Days',
            y='Carbon_Emissions_Kg',
            size='Order_ID',
            color='Customer_Rating',
            hover_name='Delivery_Status',
            title='Delivery Performance vs Carbon Emissions',
            labels={
                'Actual_Delivery_Days': 'Actual Delivery Days',
                'Carbon_Emissions_Kg': 'Average CO₂ Emissions (kg)',
                'Customer_Rating': 'Customer Rating'
            }
        )
        return fig
    
    def plot_cost_analysis(self):
        """Plot cost analysis using your cost_breakdown data"""
        cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 
                       'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
        
        available_cost_cols = [col for col in cost_columns if col in self.data.columns]
        
        if not available_cost_cols:
            return self._create_empty_plot("Cost breakdown data not available")
        
        # Calculate total costs by category
        total_costs = self.data[available_cost_cols].sum()
        
        fig = px.pie(
            values=total_costs.values,
            names=total_costs.index,
            title='Cost Breakdown Distribution',
            hole=0.4
        )
        return fig
    
    def plot_customer_feedback(self):
        """Plot customer feedback analysis from your customer_feedback data"""
        if 'Customer_Rating' not in self.data.columns or 'Issue_Category' not in self.data.columns:
            return self._create_empty_plot("Customer feedback data not available")
        
        feedback_analysis = self.data.groupby('Issue_Category').agg({
            'Customer_Rating': 'mean',
            'Carbon_Emissions_Kg': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        
        # Clean data - remove NaN values
        feedback_analysis = feedback_analysis[
            feedback_analysis[['Customer_Rating', 'Carbon_Emissions_Kg', 'Order_ID']].notna().all(axis=1)
        ]
        
        if feedback_analysis.empty:
            return self._create_empty_plot("No valid customer feedback data available")
        
        fig = px.bar(
            feedback_analysis,
            x='Issue_Category',
            y='Customer_Rating',
            color='Carbon_Emissions_Kg',
            title='Customer Feedback vs Carbon Emissions',
            labels={
                'Customer_Rating': 'Average Customer Rating',
                'Issue_Category': 'Issue Category',
                'Carbon_Emissions_Kg': 'Avg CO₂ Emissions (kg)'
            }
        )
        return fig
    
    def plot_reliability_metrics(self):
        """Plot reliability metrics from your delivery data"""
        if 'Promised_Delivery_Days' not in self.data.columns or 'Actual_Delivery_Days' not in self.data.columns:
            return self._create_empty_plot("Delivery reliability data not available")
        
        # Calculate reliability metrics
        self.data['Delivery_Delay'] = self.data['Actual_Delivery_Days'] - self.data['Promised_Delivery_Days']
        self.data['On_Time'] = self.data['Delivery_Delay'] <= 0
        
        reliability_by_carrier = self.data.groupby('Carrier').agg({
            'On_Time': 'mean',
            'Carbon_Emissions_Kg': 'mean',
            'Delivery_Cost_INR': 'mean'
        }).reset_index()
        
        # Clean data - remove NaN values
        reliability_by_carrier = reliability_by_carrier[
            reliability_by_carrier[['On_Time', 'Carbon_Emissions_Kg', 'Delivery_Cost_INR']].notna().all(axis=1)
        ]
        
        if reliability_by_carrier.empty:
            return self._create_empty_plot("No valid reliability data available")
        
        fig = px.scatter(
            reliability_by_carrier,
            x='On_Time',
            y='Carbon_Emissions_Kg',
            size='Delivery_Cost_INR',
            color='Carrier',
            title='Delivery Reliability vs Carbon Emissions by Carrier',
            labels={
                'On_Time': 'On-Time Delivery Rate',
                'Carbon_Emissions_Kg': 'Average CO₂ Emissions (kg)',
                'Delivery_Cost_INR': 'Average Delivery Cost (INR)'
            }
        )
        return fig
    
    def plot_emissions_trend(self):
        """Plot carbon emissions trend over time"""
        if 'Order_Date' not in self.data.columns:
            return self._create_empty_plot("Date data not available")
        
        try:
            self.data['Order_Date'] = pd.to_datetime(self.data['Order_Date'])
            daily_emissions = self.data.groupby('Order_Date')['Carbon_Emissions_Kg'].sum().reset_index()
            
            fig = px.line(
                daily_emissions, 
                x='Order_Date', 
                y='Carbon_Emissions_Kg',
                title='Daily Carbon Emissions Trend',
                labels={
                    'Carbon_Emissions_Kg': 'Total CO₂ Emissions (kg)',
                    'Order_Date': 'Date'
                }
            )
            
            # Add rolling average
            daily_emissions['Rolling_Avg'] = daily_emissions['Carbon_Emissions_Kg'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=daily_emissions['Order_Date'],
                y=daily_emissions['Rolling_Avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(dash='dash', color='red')
            ))
            
            return fig
        except:
            return self._create_empty_plot("Error processing date data")
    
    def _create_empty_plot(self, message):
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def plot_warehouse_analysis(self):
        """Plot warehouse inventory and emissions analysis"""
        try:
            warehouse_data = pd.read_csv(self.data_dir / 'warehouse_inventory.csv')
            
            # Calculate emissions intensity by location and product category
            if 'Location' in warehouse_data.columns and 'Product_Category' in warehouse_data.columns:
                warehouse_summary = warehouse_data.groupby(['Location', 'Product_Category']).agg({
                    'Current_Stock_Units': 'sum',
                    'Storage_Cost_per_Unit': 'mean'
                }).reset_index()
                
                fig = px.treemap(
                    warehouse_summary,
                    path=['Location', 'Product_Category'],
                    values='Current_Stock_Units',
                    color='Storage_Cost_per_Unit',
                    title='Warehouse Inventory Distribution by Location and Product Category',
                    labels={'Storage_Cost_per_Unit': 'Storage Cost per Unit (INR)'}
                )
                return fig
            else:
                return self._create_empty_plot("Warehouse location data not available")
                
        except:
            return self._create_empty_plot("Warehouse inventory data not available")
    
    def plot_toll_analysis(self):
        """Plot toll charges analysis"""
        if 'Toll_Charges_INR' not in self.data.columns:
            return self._create_empty_plot("Toll charges data not available")
        
        toll_analysis = self.data.groupby('Route').agg({
            'Toll_Charges_INR': 'mean',
            'Distance_KM': 'mean',
            'Carbon_Emissions_Kg': 'mean'
        }).reset_index()
        
        toll_analysis['Toll_per_KM'] = toll_analysis['Toll_Charges_INR'] / toll_analysis['Distance_KM']
        
        # Clean data - remove NaN values
        toll_analysis = toll_analysis[
            toll_analysis[['Toll_per_KM', 'Carbon_Emissions_Kg', 'Distance_KM']].notna().all(axis=1)
        ]
        
        if toll_analysis.empty:
            return self._create_empty_plot("No valid toll charges data available")
        
        fig = px.scatter(
            toll_analysis,
            x='Toll_per_KM',
            y='Carbon_Emissions_Kg',
            size='Distance_KM',
            color='Route',
            title='Toll Charges vs Carbon Emissions',
            labels={
                'Toll_per_KM': 'Toll Charges per KM (INR/km)',
                'Carbon_Emissions_Kg': 'Average CO₂ Emissions (kg)'
            }
        )
        return fig