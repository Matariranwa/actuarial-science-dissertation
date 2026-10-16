\left( \part{{\tiny {\tiny {\tiny }}}}#Bayesian Model Code.ipynb

# Bayesian Model packages
# Commented out IPython magic to ensure Python compatibility.
!pip install pgmpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
sns.set_style('whitegrid')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

Dataset and data preperation

df = pd.read_excel('/content/ORXData.xlsx')
print("\nDescriptive Statistics for ORX Data:")
print(df.describe())
from sklearn.feature_selection import mutual_info_regression

# The correlation matrix excluding the 'Date' column
correlation_matrix = df[numerical_features].corr()
# Print the correlation matrix
print(correlation_matrix)
# Calculate and visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Create the box plot with seaborn
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Use pandas melt to reshape data for seaborn
melted_df = pd.melt(df, id_vars=['Date'],
                    value_vars=['Intfraud', 'Exfraud', 'EmployPWS', 'ClientsPBP',
                               'Disaster', 'ICTFailure', 'ExecutionDP'],
                    var_name='Category', value_name='Loss Amount')

# Create the horizontal box plot
ax = sns.boxplot(x='Loss Amount', y='Category', data=melted_df, orient='h')

# Add a total loss category
df['Total'] = df[['Intfraud', 'Exfraud', 'EmployPWS', 'ClientsPBP',
                 'Disaster', 'ICTFailure', 'ExecutionDP']].sum(axis=1)
melted_total = pd.DataFrame({
    'Category': ['loss'] * len(df),
    'Loss Amount': df['Total']
})
sns.boxplot(x='Loss Amount', y='Category', data=melted_total,
            orient='h', ax=ax, color='lightgreen')

# Customize the plot
plt.title('Box Plot of Operational Risk Loss Variables')
plt.tight_layout()

# Show the plot
plt.show()


#Frequency Distributions
columns_to_plot = [col for col in df.columns if col != 'Date']
num_plots = len(columns_to_plot)
cols = 2
rows = math.ceil(num_plots / cols)
fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = axes.flatten()
for i, col in enumerate(columns_to_plot):
    ax = axes[i]
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Frequency Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    else:
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Frequency Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

!apt-get -qq update
!apt-get -qq install python3-distutils
!pip install pymc3 # Install pymc3 if not already installed
!pip install --upgrade setuptools
!pip install --upgrade numpy
!pip install --upgrade theano
!pip install --upgrade setuptools wheel
!pip install --upgrade pymc numpy scipy pandas arviz

# Define Bayesian Model
with pm.Model() as bayesian_model:
    # Priors:
    risk_priors = {
        col: pm.Beta(f"{col}_prior", alpha=np.random.uniform(1, 5), beta=np.random.uniform(1, 5)) for col in risk_categories
    }

# Likelihood: Bernoulli distribution based on observed risk category presence
    risk_obs = {
        col: pm.Bernoulli(f"{col}_obs", p=risk_priors[col], observed=(df[col] > 0).astype(int))
        for col in risk_categories
    }

# Run inference (sampling)
    trace = pm.sample(2000, return_inferencedata=True)

# Custom Styling for Posterior Plots
fig, axes = plt.subplots(len(risk_categories), 1, figsize=(8, len(risk_categories) * 3))
colors = ['#E67E22', '#3498DB', '#27AE60', '#8E44AD', '#F1C40F', '#D35400', '#2C3E50']  # Professional colors

for i, col in enumerate(risk_categories):
    az.plot_posterior(trace, var_names=[f"{col}_prior"], ax=axes[i], hdi_prob=0.94, color=colors[i % len(colors)])
    axes[i].set_title(f'Posterior Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i].grid(True, linestyle="--", alpha=0.6)

# Plot Results
plt.tight_layout()
plt.show()

# Define priors for each column in the dataset with distinct distributions
with pm.Model() as model:
    # Define priors for each distribution
    priors = {}
    for col in df.columns:
        if np.random.rand() > 0.5:  # Randomly decide whether to use Beta or Normal prior
            priors[col] = pm.Beta(f'prior_{col}', alpha=2, beta=2)
        else:
            priors[col] = pm.Normal(f'prior_{col}', mu=0, sigma=1)

    # Define likelihoods for each column (using different priors for each column)
    likelihoods = {}
    for i, col in enumerate(df.columns):
        if isinstance(priors[col], pm.Beta):
            likelihoods[col] = pm.Binomial(f'like_{col}', n=n, p=priors[col], observed=obs_values[i])
        else:
            # For Normal prior, assume normally distributed observations
            likelihoods[col] = pm.Normal(f'like_{col}', mu=priors[col], sigma=1, observed=obs_values[i])

    # Define comparisons between variables
    pm.Deterministic('difference', priors[df.columns[1]] - priors[df.columns[0]])
    pm.Deterministic('relation', (priors[df.columns[1]] / priors[df.columns[0]]) - 1)

    # Sample from posterior
    trace = pm.sample(draws=50000, step=pm.Metropolis(), progressbar=True, cores=2)

# Plot Results
az.plot_trace(trace)
plt.tight_layout()
plt.show()

Causal Discovery

import tigramite.independence_tests
print(dir(tigramite.independence_tests))
from tigramite.independence_tests.parcorr import ParCorr

# Our Data must be array of shape (time, variables)
print(data.shape)
dataframe = pp.DataFrame(data)
cond_ind_test = ParCorr()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
results = pcmci.run_pcmci(tau_max=5, pc_alpha=None)
pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                     val_matrix=results['val_matrix'],
                                     alpha_level=0.05)
# Convert DataFrame to Tigramite DataFrame
dataframe = pp.DataFrame(df.values, var_names=df.columns)

# Perform PCMCI (Partial Correlation-based PCMCI)
cond_ind_test = ParCorr()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
results = pcmci.run_pcmci(tau_max=5, pc_alpha=None)

# Print significant links
pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                     val_matrix=results['val_matrix'],
                                     alpha_level=0.05)

# Get the link_matrix from the PCMCI results
link_matrix = results['graph']  # or results['p_matrix'] if you want p-values

# Plot the causal graph directly using the link_matrix
tp.plot_time_series_graph(
    figsize=(6, 3),
    val_matrix=results['val_matrix'], # Use the val_matrix from PCMCI results
    graph=link_matrix,  # Use the link_matrix directly
    var_names=df.columns,
    link_colorbar_label=''
)

# Get causal adjacency matrix
adj_matrix = model.adjacency_matrix_
# Construct a causal graph
import networkx as nx
G = nx.DiGraph()
columns = data_scaled.columns
for i, col_i in enumerate(columns):
    for j, col_j in enumerate(columns):
        if adj_matrix[i, j] != 0:
            G.add_edge(col_j, col_i)
            for node in list(G.nodes):
    if node != "TotalLoss" and not G.has_edge(node, "TotalLoss"):
        G.add_edge(node, "TotalLoss")

# Visualize the updated causal graph
plt.figure(figsize=(10, 7))
nx.draw(G, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
plt.show()

Conditional Probabilities and Bayesian model output

# Define risk categories
risk_categories = ['Intfraud', 'Exfraud', 'EmployPWS', 'ClientsPBP', 'Disaster', 'ICTFailure', 'ExecutionDP']

# FIXED: Completely revised categorization approach using statistical thresholds
def meaningful_categorize(series):
    # Use mean and standard deviation for more meaningful thresholds
    mean_val = series.mean()
    std_val = series.std()

    # Define thresholds: Low (below mean-0.5*std), High (above mean+0.5*std), Medium (in between)
    low_threshold = mean_val - 0.5 * std_val
    high_threshold = mean_val + 0.5 * std_val

    categories = []
    for val in series:
        if val <= low_threshold:
            categories.append('Low')
        elif val >= high_threshold:
            categories.append('High')
        else:
            categories.append('Medium')

    # Calculate actual percentages
    cat_series = pd.Series(categories)
    total = len(categories)
    percentage = {
        'Low': (cat_series == 'Low').sum() / total * 100,
        'Medium': (cat_series == 'Medium').sum() / total * 100,
        'High': (cat_series == 'High').sum() / total * 100
    }

    return categories, percentage

# Dictionary to store actual percentages for each category
actual_percentages = {}

# Create categorical variables with truly meaningful thresholds
for cat in risk_categories:
    df[f'{cat}_cat'], actual_percentages[cat] = meaningful_categorize(df[cat])

# Calculate total operational risk (sum of all categories)
df['TotalRisk'] = df[risk_categories].sum(axis=1)
df['TotalRisk_cat'], actual_percentages['TotalRisk'] = meaningful_categorize(df['TotalRisk'])

# Calculate correlations
correlation_matrix = df[risk_categories].corr()

# Define LinearRegression class
class LinearRegression:
    def fit(self, X, y):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        self.coef_ = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
        return self

    def predict(self, X):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ np.concatenate([[self.intercept_], self.coef_])

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v

# Calculate impact percentages using regression analysis
def calculate_impact(X, y):
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    # Calculate contributions based on standardized coefficients
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    std_model = LinearRegression()
    std_model.fit(X_std, y)

    # Calculate absolute standardized coefficients
    abs_std_coef = np.abs(std_model.coef_)

    # Calculate contributions
    contributions = abs_std_coef / np.sum(abs_std_coef) * 100

    # Calculate p-values
    n = X.shape[0]
    k = X.shape[1]
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - k - 1)

    # Calculate standard errors
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se = np.sqrt(np.diag(cov_matrix)[1:])

    # Calculate t-statistics and p-values
    t_stats = model.coef_ / se

    # Two-tailed p-values
    p_values = []
    for t in t_stats:
        p = 2 * (1 - stats.t.cdf(abs(t), n - k - 1))
        p_values.append(p)

    return contributions, p_values, r2

# Prepare data for impact calculation
X = df[risk_categories].values
y = df['TotalRisk'].values
contributions, p_values, r2 = calculate_impact(X, y)

# Create visualization of the Bayesian Network
plt.figure(figsize=(14, 10))
ax = plt.gca()
ax.set_facecolor('#f8f8f8')

# Node positions
positions = {
    'Intfraud': (1, 8),
    'Exfraud': (4, 8),
    'ICTFailure': (7, 8),
    'EmployPWS': (1, 5),
    'ExecutionDP': (7, 5),
    'ClientsPBP': (2.5, 2),
    'Disaster': (5.5, 2),
    'TotalRisk': (4, 5)
}

# Node colors
node_colors = {
    'Intfraud': '#90ee90',
    'Exfraud': '#90ee90',
    'ICTFailure': '#90ee90',
    'EmployPWS': '#ff9090',
    'ExecutionDP': '#ff9090',
    'ClientsPBP': '#ff9090',
    'Disaster': '#ff9090',
    'TotalRisk': '#e0e0ff'
}

# Draw nodes
node_objects = {}
for node, pos in positions.items():
    color = node_colors[node]
    if node == 'TotalRisk':
        label = 'Operational Risk'
        width, height = 2, 1
    else:
        labels = {
            'Intfraud': 'Internal Fraud',
            'Exfraud': 'External Fraud',
            'EmployPWS': 'Employment Practices\n& Workplace Safety',
            'ClientsPBP': 'Clients, Products &\nBusiness Practices',
            'Disaster': 'Disaster',
            'ICTFailure': 'ICT Failure',
            'ExecutionDP': 'Execution, Delivery &\nProcess Management'
        }
        label = labels[node]
        if node in ['Intfraud', 'Exfraud', 'ICTFailure']:
            width, height = 1.8, 1.2
        else:
            width, height = 1.8, 1

    rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height,
                    facecolor=color, edgecolor='black', alpha=0.9,
                    linewidth=1)
    node_objects[node] = rect
    ax.add_patch(rect)

    # Add title
    plt.text(pos[0], pos[1] + 0.2, label,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=10, fontweight='bold')

    # Percentages for categorical variables
    if node != 'TotalRisk':
        contribution = contributions[risk_categories.index(node)]
        text_y = pos[1]
        if node in ['Intfraud', 'Exfraud', 'ICTFailure']:
            text_y = pos[1] - 0.1
            plt.text(pos[0], text_y, f"Contribution: {contribution:.1f}%",
                     horizontalalignment='center', fontsize=9)

            # Add small histogram for top nodes
            hist_width = 0.8
            hist_height = 0.3
            hist_data = df[node].values
            hist_bins = min(10, len(set(hist_data)))
            hist_y = pos[1] - 0.3
            ax.axhline(y=hist_y, xmin=(pos[0]-hist_width/2-0.5)/12, xmax=(pos[0]+hist_width/2+0.5)/12,
                       color='black', linewidth=1)
            ax.axvline(x=pos[0]-hist_width/2, ymin=(hist_y-hist_height-0.5)/10, ymax=(hist_y+0.5)/10,
                       color='black', linewidth=1)

            # Normalize histogram
            counts, bin_edges = np.histogram(hist_data, bins=hist_bins)
            counts = counts / max(counts) * hist_height
            bin_width = hist_width / hist_bins

            # Histogram bars
            for i, count in enumerate(counts):
                bar_height = count
                bar_x = pos[0] - hist_width/2 + i * bin_width + bin_width/2
                ax.add_patch(Rectangle((bar_x-bin_width/2, hist_y-bar_height),
                                      bin_width*0.8, bar_height,
                                      facecolor='#b0b0ff',
                                      edgecolor='black', linewidth=0.5))
        else:
            # Add actual percentages for categorical values
            perc = actual_percentages[node]
            text_y = pos[1] - 0.1
            plt.text(pos[0], text_y, f"High: {perc.get('High', 0):.1f}%",
                     horizontalalignment='center', fontsize=9)
            plt.text(pos[0], text_y-0.2, f"Medium: {perc.get('Medium', 0):.1f}%",
                     horizontalalignment='center', fontsize=9)
            plt.text(pos[0], text_y-0.4, f"Low: {perc.get('Low', 0):.1f}%",
                     horizontalalignment='center', fontsize=9)
    else:
        # Add actual percentages for Total Risk
        perc = actual_percentages[node]
        text_y = pos[1] - 0.1
        plt.text(pos[0], text_y, f"High: {perc.get('High', 0):.1f}%",
                 horizontalalignment='center', fontsize=9)
        plt.text(pos[0], text_y-0.2, f"Medium: {perc.get('Medium', 0):.1f}%",
                 horizontalalignment='center', fontsize=9)
        plt.text(pos[0], text_y-0.4, f"Low: {perc.get('Low', 0):.1f}%",
                 horizontalalignment='center', fontsize=9)

# Draw arrows between nodes
for i, source in enumerate(risk_categories):
    # Calculate impact and significance
    impact = contributions[i]
    p_value = p_values[i]

    # Get positions
    source_pos = positions[source]
    target_pos = positions['TotalRisk']

    # Calculate midpoint for label
    mid_x = (source_pos[0] + target_pos[0]) / 2
    mid_y = (source_pos[1] + target_pos[1]) / 2

    # Adjust midpoint to avoid overlap
    offset_x = 0.3 if source_pos[0] < target_pos[0] else -0.3
    offset_y = 0.3 if source_pos[1] < target_pos[1] else -0.3

    # Draw arrow
    if source in ['EmployPWS', 'ExecutionDP']:
        # Horizontal arrows
        if source == 'EmployPWS':
            start_x = source_pos[0] + 0.9
            end_x = target_pos[0] - 1
            y = source_pos[1]
            plt.arrow(start_x, y, end_x-start_x-0.1, 0,
                     head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
        else:
            start_x = source_pos[0] - 0.9
            end_x = target_pos[0] + 1
            y = source_pos[1]
            plt.arrow(start_x, y, end_x-start_x+0.1, 0,
                     head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    else:
        # Draw other arrows
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        length = np.sqrt(dx**2 + dy**2)
        norm_dx = dx / length
        norm_dy = dy / length

        # Shorten the arrow at both ends
        start_x = source_pos[0] + norm_dx * 0.5
        start_y = source_pos[1] + norm_dy * 0.5
        end_x = target_pos[0] - norm_dx * 0.5
        end_y = target_pos[1] - norm_dy * 0.5

        plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y,
                 head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)

    # Add impact label with white background
    label_x = mid_x + offset_x
    label_y = mid_y + offset_y

    # Create white background for label
    rect = Rectangle((label_x-0.4, label_y-0.2), 0.8, 0.4, facecolor='white',
                    edgecolor='black', alpha=0.9, linewidth=1)
    ax.add_patch(rect)

    # Add p-value text
    if p_value < 0.001:
        p_text = "p<0.001"
    else:
        p_text = f"p={p_value:.3f}"

    plt.text(label_x, label_y, f"Impact: {impact:.1f}%",
             horizontalalignment='center', fontsize=9)
    plt.text(label_x, label_y-0.15, p_text,
             horizontalalignment='center', fontsize=9)

# Correlations as dashed lines
corr_thresholds = {
    ('Intfraud', 'Exfraud'): 0.29,
    ('Exfraud', 'ICTFailure'): 0.31,
    ('EmployPWS', 'ClientsPBP'): 0.47,
    ('ExecutionDP', 'ClientsPBP'): 0.64
}

for (source, target), corr in corr_thresholds.items():
    source_pos = positions[source]
    target_pos = positions[target]

    # Calculate connection points
    if source in ['Intfraud', 'Exfraud'] and target in ['Exfraud', 'ICTFailure']:
        # Horizontal connection for top nodes
        plt.plot([source_pos[0] + 0.9, target_pos[0] - 0.9],
                [source_pos[1], target_pos[1]],
                'b--', linewidth=1, alpha=0.7)
        # Add correlation value
        mid_x = (source_pos[0] + target_pos[0]) / 2
        plt.text(mid_x, source_pos[1] - 0.1, f"r={corr}",
                color='blue', fontsize=8,
                horizontalalignment='center')
    else:
        # Diagonal connections
        plt.plot([source_pos[0], target_pos[0]],
                [source_pos[1] - 0.5, target_pos[1] + 0.5],
                'b--', linewidth=1, alpha=0.7)
        # Add correlation value
        mid_x = (source_pos[0] + target_pos[0]) / 2
        mid_y = (source_pos[1] + target_pos[1]) / 2
        plt.text(mid_x, mid_y, f"r={corr}",
                color='blue', fontsize=8,
                horizontalalignment='center')

# Add title
plt.title('Operational Risk Causal Bayesian Network', fontsize=14, fontweight='bold')

# Remove axis
plt.axis('off')
plt.tight_layout()

# Show summary
print("Operational Risk Bayesian Network Analysis")
print("-----------------------------------------")
print(f"Overall model fit (R²): {r2:.4f}")
print("\nRisk Factor Contributions:")
for i, cat in enumerate(risk_categories):
    print(f"{cat}: {contributions[i]:.1f}% (p-value: {p_values[i]:.4f})")

print("\nRisk Level Distributions:")
for cat in risk_categories + ['TotalRisk']:
    perc = actual_percentages[cat]
    print(f"{cat}: Low={perc.get('Low', 0):.1f}%, Medium={perc.get('Medium', 0):.1f}%, High={perc.get('High', 0):.1f}%")

print("\nSignificant Correlations:")
for (source, target), corr in corr_thresholds.items():
    print(f"{source} - {target}: r={corr}")

plt.savefig('operational_risk_bayesian_network.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Create a heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Operational Risk Factors', fontsize=14)
plt.tight_layout()
plt.savefig('operational_risk_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()