import streamlit as st
from carbon_model import CarbonModel
from carbon_visualizations import CarbonVisualizations
from carbon_recommendations import CarbonRecommendations

def main():
    st.set_page_config(
        page_title="Carbon Sustainability Tracker",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌿 Carbon Sustainability Tracker - NexGen Logistics")
    st.markdown("### Transforming Logistics Operations through Carbon Intelligence")
    
    # Initialize components
    model = CarbonModel()
    visualizations = CarbonVisualizations(model)
    recommendations = CarbonRecommendations(model)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        [
            "Model Training & Prediction", 
            "Data Visualizations", 
            "Sustainability Recommendations",
            "Real-time Calculator"
        ]
    )
    
    if app_mode == "Model Training & Prediction":
        show_model_page(model)
    elif app_mode == "Data Visualizations":
        show_visualizations_page(visualizations)
    elif app_mode == "Sustainability Recommendations":
        show_recommendations_page(recommendations)
    elif app_mode == "Real-time Calculator":
        show_calculator_page(model)

def show_model_page(model):
    st.header("🤖 Carbon Prediction Model")
    
    tab1, tab2, tab3 = st.tabs(["Model Training", "Make Predictions", "Model Performance"])
    
    with tab1:
        show_model_training(model)
    
    with tab2:
        show_prediction_interface(model)
    
    with tab3:
        show_model_performance(model)

def show_model_training(model):
    st.subheader("Train Carbon Prediction Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Model Overview
        This machine learning model predicts carbon emissions based on your logistics data:
        - Distance traveled
        - Vehicle characteristics  
        - Fuel consumption
        - Route parameters
        - Order details
        """)
        
        # Model configuration
        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "Gradient Boosting", "Linear Regression"]
        )
        
        train_size = st.slider("Training Data Size (%)", 70, 90, 80)
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model with your dataset... This may take a few moments."):
                performance = model.train_model(model_type, train_size)
                
            if performance:
                st.success("Model trained successfully on your data!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{performance['r2']:.3f}")
                with col2:
                    st.metric("MSE", f"{performance['mse']:.2f}")
                with col3:
                    st.metric("MAE", f"{performance['mae']:.2f}")
    
    with col2:
        st.subheader("Model Status")
        if model.is_trained():
            st.success("✅ Model Ready")
            st.metric("Features Used", len(model.get_feature_names()))
            st.metric("Training Samples", model.get_training_info()['n_samples'])
        else:
            st.warning("⚠️ Model Not Trained")
            st.info("Click 'Train Model' to start training with your data")

def show_prediction_interface(model):
    st.subheader("Predict Carbon Emissions for New Orders")
    
    if not model.is_trained():
        st.warning("Please train the model first using the 'Model Training' tab.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Order Parameters")
        distance = st.number_input("Distance (KM)", min_value=1.0, value=150.0, step=10.0)
        order_value = st.number_input("Order Value (INR)", min_value=1.0, value=5000.0, step=100.0)
        product_category = st.selectbox("Product Category", model.get_categories('Product_Category'))
        priority = st.selectbox("Delivery Priority", model.get_categories('Priority'))
        
    with col2:
        st.markdown("### Vehicle Parameters")
        vehicle_type = st.selectbox("Vehicle Type", model.get_categories('Vehicle_Type'))
        fuel_efficiency = st.number_input("Fuel Efficiency (KM/L)", min_value=1.0, value=8.0, step=0.5)
        fuel_consumption = st.number_input("Fuel Consumption (L)", min_value=1.0, value=50.0, step=5.0)
        capacity = st.number_input("Vehicle Capacity (KG)", min_value=100.0, value=2000.0, step=100.0)
    
    # Route factors
    st.markdown("### Route Factors")
    col1, col2, col3 = st.columns(3)
    with col1:
        traffic_delay = st.number_input("Traffic Delay (Minutes)", min_value=0, value=15, step=5)
    with col2:
        weather_impact = st.selectbox("Weather Impact", ["None", "Light_Rain", "Heavy_Rain", "Fog"])
    with col3:
        toll_charges = st.number_input("Toll Charges (INR)", min_value=0.0, value=150.0, step=50.0)
    
    if st.button("Predict Carbon Emissions", type="primary"):
        # Prepare input data
        input_data = {
            'Distance_KM': distance,
            'Order_Value_INR': order_value,
            'Product_Category': product_category,
            'Priority': priority,
            'Vehicle_Type': vehicle_type,
            'Fuel_Efficiency_KM_per_L': fuel_efficiency,
            'Fuel_Consumption_L': fuel_consumption,
            'Capacity_KG': capacity,
            'Traffic_Delay_Minutes': traffic_delay,
            'Weather_Impact': weather_impact,
            'Toll_Charges_INR': toll_charges
        }
        
        # Make prediction
        prediction = model.predict_single(input_data)
        
        if prediction:
            st.success(f"### Predicted Carbon Emissions: {prediction['prediction']:.2f} kg CO₂")
            
            # Show comparison
            avg_emission = model.get_average_emission()
            efficiency = "Better" if prediction['prediction'] < avg_emission else "Worse"
            percentage_diff = ((prediction['prediction'] - avg_emission) / avg_emission) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Emission", f"{avg_emission:.2f} kg")
            with col2:
                st.metric("Your Prediction", f"{prediction['prediction']:.2f} kg")
            with col3:
                st.metric("Comparison", f"{efficiency} by {abs(percentage_diff):.1f}%")
            
            # Show feature impact
            st.subheader("Feature Impact on Prediction")
            impact_df = model.get_feature_impact(input_data)
            st.dataframe(impact_df)

def show_model_performance(model):
    st.subheader("Model Performance Analysis")
    
    if not model.is_trained():
        st.warning("Please train the model first using the 'Model Training' tab.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance
        st.plotly_chart(model.plot_feature_importance(), use_container_width=True)
        
        # Prediction vs Actual
        st.plotly_chart(model.plot_prediction_vs_actual(), use_container_width=True)
    
    with col2:
        # Error distribution
        st.plotly_chart(model.plot_error_distribution(), use_container_width=True)
        
        # Model comparison
        st.plotly_chart(model.plot_model_comparison(), use_container_width=True)

def show_visualizations_page(visualizations):
    st.header("📊 Data Visualizations & Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Carbon Overview", "Vehicle Analysis", "Route Analysis", "Performance Metrics"
    ])
    
    with tab1:
        show_carbon_overview(visualizations)
    
    with tab2:
        show_vehicle_analysis(visualizations)
    
    with tab3:
        show_route_analysis(visualizations)
    
    with tab4:
        show_performance_metrics(visualizations)

def show_carbon_overview(visualizations):
    st.subheader("Carbon Emissions Overview")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        carrier_filter = st.selectbox("Filter by Carrier", ["All"] + visualizations.get_carriers())
    with col2:
        vehicle_filter = st.selectbox("Filter by Vehicle Type", ["All"] + visualizations.get_vehicle_types())
    with col3:
        product_filter = st.selectbox("Filter by Product", ["All"] + visualizations.get_product_categories())
    
    # Key metrics
    metrics = visualizations.get_carbon_metrics(carrier_filter, vehicle_filter, product_filter)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CO₂", f"{metrics['total_co2']:,.0f} kg")
    with col2:
        st.metric("Avg per Order", f"{metrics['avg_per_order']:.1f} kg")
    with col3:
        st.metric("Carbon Intensity", f"{metrics['intensity']:.3f} kg/INR")
    with col4:
        st.metric("Efficiency Score", f"{metrics['efficiency_score']:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(visualizations.plot_emissions_by_vehicle(), use_container_width=True)
        st.plotly_chart(visualizations.plot_carrier_comparison(), use_container_width=True)
    with col2:
        st.plotly_chart(visualizations.plot_product_emissions(), use_container_width=True)
        st.plotly_chart(visualizations.plot_priority_impact(), use_container_width=True)

def show_vehicle_analysis(visualizations):
    st.subheader("Vehicle Fleet Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(visualizations.plot_vehicle_efficiency(), use_container_width=True)
        st.plotly_chart(visualizations.plot_fuel_consumption_analysis(), use_container_width=True)
    with col2:
        st.plotly_chart(visualizations.plot_capacity_analysis(), use_container_width=True)
        st.plotly_chart(visualizations.plot_emissions_by_location(), use_container_width=True)

def show_route_analysis(visualizations):
    st.subheader("Route Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(visualizations.plot_route_efficiency(), use_container_width=True)
        st.plotly_chart(visualizations.plot_weather_impact(), use_container_width=True)
    with col2:
        st.plotly_chart(visualizations.plot_traffic_impact(), use_container_width=True)
        st.plotly_chart(visualizations.plot_distance_vs_emissions(), use_container_width=True)
    
    # Route optimization
    st.subheader("Route Optimization Opportunities")
    high_impact_routes = visualizations.get_high_impact_routes()
    st.dataframe(high_impact_routes)

def show_performance_metrics(visualizations):
    st.subheader("Delivery Performance Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(visualizations.plot_delivery_performance(), use_container_width=True)
        st.plotly_chart(visualizations.plot_cost_analysis(), use_container_width=True)
    with col2:
        st.plotly_chart(visualizations.plot_customer_feedback(), use_container_width=True)
        st.plotly_chart(visualizations.plot_reliability_metrics(), use_container_width=True)

def show_recommendations_page(recommendations):
    st.header("💡 Sustainability Recommendations")
    
    tab1, tab2, tab3 = st.tabs([
        "Action Plan", "Cost-Benefit Analysis", "Implementation Roadmap"
    ])
    
    with tab1:
        show_action_plan(recommendations)
    
    with tab2:
        show_cost_benefit_analysis(recommendations)
    
    with tab3:
        show_implementation_roadmap(recommendations)

def show_action_plan(recommendations):
    st.subheader("Sustainability Action Plan")
    
    # Overall impact
    impact = recommendations.calculate_overall_impact()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Potential CO₂ Reduction", f"{impact['reduction_potential']:.1f}%")
    with col2:
        st.metric("Cost Savings", f"₹{impact['cost_savings']:,.0f}")
    with col3:
        st.metric("Implementation Period", f"{impact['timeline']} months")
    
    # Recommendations
    st.subheader("Priority Recommendations")
    recs = recommendations.get_priority_recommendations()
    
    for i, rec in enumerate(recs, 1):
        with st.expander(f"#{i} {rec['title']} - Impact: {rec['impact']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Expected Reduction:** {rec['reduction']}")
                st.write(f"**Implementation Cost:** {rec['cost']}")
            with col2:
                st.metric("ROI", rec['roi'])
                progress = st.progress(rec['feasibility'] / 100)
                st.caption(f"Feasibility: {rec['feasibility']}%")
            
            if st.button(f"Implement {rec['title']}", key=f"imp_{i}"):
                st.success(f"Implementation plan for {rec['title']} started!")

def show_cost_benefit_analysis(recommendations):
    st.subheader("Cost-Benefit Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(recommendations.plot_cost_benefit(), use_container_width=True)
        st.plotly_chart(recommendations.plot_roi_analysis(), use_container_width=True)
    with col2:
        st.plotly_chart(recommendations.plot_savings_timeline(), use_container_width=True)
        st.plotly_chart(recommendations.plot_implementation_priority(), use_container_width=True)

def show_implementation_roadmap(recommendations):
    st.subheader("Implementation Roadmap")
    
    st.plotly_chart(recommendations.plot_implementation_timeline(), use_container_width=True)
    
    # Phase details
    phases = recommendations.get_implementation_phases()
    for phase in phases:
        with st.expander(f"Phase {phase['phase']}: {phase['name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Duration:** {phase['duration']}")
                st.write(f"**Key Activities:**")
                for activity in phase['activities']:
                    st.write(f"- {activity}")
            with col2:
                st.write(f"**Resources Required:** {phase['resources']}")
                st.write(f"**Success Metrics:** {phase['metrics']}")

def show_calculator_page(model):
    st.header("🧮 Real-time Carbon Calculator")
    
    st.markdown("""
    Calculate carbon emissions for new logistics operations using your actual data patterns.
    This tool helps estimate environmental impact before making delivery decisions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Parameters")
        distance = st.number_input("Distance (KM)", min_value=1.0, value=100.0, step=10.0, key="calc_distance")
        vehicle_type = st.selectbox("Vehicle Type", model.get_categories('Vehicle_Type'), key="calc_vehicle")
        fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "Electric", "CNG", "Hybrid"])
        load_factor = st.slider("Load Factor (%)", 0, 100, 75)
        
    with col2:
        st.subheader("Route Factors")
        traffic = st.select_slider("Traffic Conditions", 
                                 options=["Free Flow", "Light", "Moderate", "Heavy", "Congested"],
                                 value="Moderate")
        weather = st.select_slider("Weather Conditions",
                                 options=["Clear", "Light Rain", "Heavy Rain", "Fog", "Extreme"],
                                 value="Clear")
        route_type = st.selectbox("Route Type", ["Highway", "Urban", "Mixed", "Rural"])
        
    if st.button("Calculate Carbon Impact", type="primary"):
        # Calculate emissions
        calculation = model.calculate_realtime_emissions(
            distance, vehicle_type, fuel_type, load_factor,
            traffic, weather, route_type
        )
        
        # Display results
        st.subheader("📊 Calculation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CO₂ Emissions", f"{calculation['co2_emissions']:.2f} kg")
        with col2:
            st.metric("Fuel Consumption", f"{calculation['fuel_used']:.2f} L")
        with col3:
            st.metric("Cost Impact", f"₹{calculation['cost_impact']:.2f}")
        with col4:
            efficiency_label = "High" if calculation['efficiency_score'] > 0.7 else "Medium" if calculation['efficiency_score'] > 0.4 else "Low"
            st.metric("Efficiency", efficiency_label)

if __name__ == "__main__":
    main()