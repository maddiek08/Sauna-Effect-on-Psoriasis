import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def sauna_schedule(t):
    """Simulates sauna use 3 times per week for 30 minutes per session over 4 weeks."""
    session_minutes = 30 / (24 * 60)  # Convert 30 minutes to fraction of a day
    return session_minutes if (t % 7) in [0, 2, 4] else 0  # Sauna on days 0, 2, 4 of each week
    """Simulates sauna use 3 times per week by applying a periodic function."""
    return 1 if (t % 7) in [0, 2, 4] else 0  # Sauna on days 0, 2, 4 of each week

def psoriasis_model(y, t):
    """
    Differential equation model for psoriasis progression with sauna therapy.
    y[0] = inflammation level (cytokine concentration)
    y[1] = skin hydration level
    y[2] = keratinocyte proliferation rate
    y[3] = cortisol level
    """
    infl, hydration, prolif, cortisol = y
    
    # Parameters (hypothetical values)
    k_infl = 0.1  # Inflammation decay rate
    k_hydr = 0.05  # Hydration loss rate
    k_prolif = 0.08  # Keratinocyte proliferation rate
    k_cortisol = 0.07  # Cortisol decay rate
    
    sauna_effect = 0.5  # Generalized sauna effect on inflammation
    heat_exposure = sauna_schedule(t)  # Only active on sauna days
    
    # Differential equations
    d_infl_dt = -k_infl * infl * sauna_effect * heat_exposure - 0.05 * cortisol  # Cortisol helps reduce inflammation
    d_hydr_dt = k_hydr * sauna_effect * heat_exposure - k_hydr * hydration - 0.02 * heat_exposure  # Sweat loss
    d_prolif_dt = -k_prolif * sauna_effect * heat_exposure * prolif
    d_cortisol_dt = -k_cortisol * cortisol + 0.1 * heat_exposure  # Sauna increases cortisol transiently
    
    return [d_infl_dt, d_hydr_dt, d_prolif_dt, d_cortisol_dt]

# Initial conditions
initial_conditions = [1.0, 0.5, 1.5, 1.0]  # Inflammation, hydration, proliferation, cortisol

time = np.linspace(0, 28, 100)  # Simulate over 4 weeks (28 days)  # Simulate over 30 days

# Solve ODEs
results = odeint(psoriasis_model, initial_conditions, time)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(time, results[:, 0], label='Inflammation')
plt.plot(time, results[:, 1], label='Skin Hydration')
plt.plot(time, results[:, 2], label='Keratinocyte Proliferation')
plt.plot(time, results[:, 3], label='Cortisol', linestyle='dashed')
plt.xlabel('Time (days)')
plt.ylabel('Normalized Values')
plt.legend()
plt.title('Psoriasis Response to Sauna Therapy')
plt.show()

# Machine Learning Model
# Create a mock dataset since the CSV file is unavailable
data = pd.DataFrame({
    "Heat_Exposure": np.random.choice([0, 1], size=30),
    "Hydration": np.random.uniform(0.3, 1.0, size=30),
    "PASI_Baseline": np.random.uniform(5, 15, size=30),
    "Cytokine_Levels": np.random.uniform(1, 10, size=30),
    "Cortisol_Levels": np.random.uniform(0.5, 2.5, size=30),
    "PASI_Reduction": np.random.uniform(0, 5, size=30)
})

# Filter for participants (15 males, 15 females, PASI-balanced, no medication)
data = data.sample(30)  # Simulating an equal split

# Features: Heat exposure, hydration, PASI score, cytokine levels, cortisol levels
X = data[["Heat_Exposure", "Hydration", "PASI_Baseline", "Cytokine_Levels", "Cortisol_Levels"]]
y = data["PASI_Reduction"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
predictions = rf_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
