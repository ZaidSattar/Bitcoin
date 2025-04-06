import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def view_results_matplotlib():
    """Display all images in the output/figures directory using matplotlib"""
    figures_dir = "output/figures"
    fig_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    
    # Calculate grid dimensions
    n_images = len(fig_files)
    n_cols = 2
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    # Add each image to the figure
    for i, fig_file in enumerate(fig_files):
        img = imread(os.path.join(figures_dir, fig_file))
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.imshow(img)
        ax.set_title(fig_file.replace('.png', '').replace('_', ' ').title())
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def view_results_tkinter():
    """Display a simple UI to view figures one at a time"""
    figures_dir = "output/figures"
    fig_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    
    root = tk.Tk()
    root.title("Bitcoin Price Prediction Results")
    root.geometry("1000x700")
    
    # Frame for controls
    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(fill=tk.X)
    
    current_index = tk.IntVar(value=0)
    
    # Label to display image name
    title_label = ttk.Label(control_frame, text="", font=("Arial", 16))
    title_label.pack(pady=10)
    
    # Frame for image
    image_frame = ttk.Frame(root)
    image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    image_label = ttk.Label(image_frame)
    image_label.pack(fill=tk.BOTH, expand=True)
    
    def update_image():
        idx = current_index.get()
        fig_file = fig_files[idx]
        title = fig_file.replace('.png', '').replace('_', ' ').title()
        title_label.config(text=title)
        
        img = Image.open(os.path.join(figures_dir, fig_file))
        
        # Resize image to fit the window
        window_width = image_frame.winfo_width()
        window_height = image_frame.winfo_height()
        
        # Initial values for first render
        if window_width <= 1:
            window_width = 900
        if window_height <= 1:
            window_height = 600
            
        # Calculate new dimensions keeping aspect ratio
        img_width, img_height = img.size
        ratio = min(window_width/img_width, window_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference
    
    def next_image():
        current_index.set((current_index.get() + 1) % len(fig_files))
        update_image()
    
    def prev_image():
        current_index.set((current_index.get() - 1) % len(fig_files))
        update_image()
    
    # Buttons for navigation
    button_frame = ttk.Frame(control_frame)
    button_frame.pack(pady=10)
    
    prev_button = ttk.Button(button_frame, text="Previous", command=prev_image)
    prev_button.pack(side=tk.LEFT, padx=10)
    
    next_button = ttk.Button(button_frame, text="Next", command=next_image)
    next_button.pack(side=tk.LEFT, padx=10)
    
    # Initialize the first image after a short delay to let the window render
    root.after(100, update_image)
    
    root.mainloop()

def view_insights():
    """Display the insights summary"""
    with open("output/insights.txt", "r") as f:
        insights = f.read()
    
    root = tk.Tk()
    root.title("Bitcoin Price Prediction Insights")
    root.geometry("800x600")
    
    text_widget = tk.Text(root, wrap=tk.WORD, padx=20, pady=20, font=("Courier", 12))
    text_widget.pack(fill=tk.BOTH, expand=True)
    text_widget.insert(tk.END, insights)
    text_widget.config(state=tk.DISABLED)
    
    root.mainloop()

def parse_insights_txt():
    """Parse the insights.txt file into a structured dictionary"""
    insights_data = {}
    current_section = None
    
    with open("output/insights.txt", "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith('INSIGHTS'):
            # Start of a new section
            current_section = line.replace(' INSIGHTS', '').lower()
            insights_data[current_section] = {}
            i += 2  # Skip the ========== line
        elif ':' in line and current_section:
            # Key-value pair
            key, value_str = line.split(':', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Handle specific known types
            if key == 'cluster_sizes':
                try:
                    # Parse dictionary format like {1: 30, 0: 24, 3: 23, 2: 22}
                    value_str = value_str.strip('{}')
                    pairs = value_str.split(',')
                    value = {}
                    for pair in pairs:
                        if ':' in pair:
                            k, v = pair.split(':')
                            value[k.strip()] = int(v.strip())
                except Exception as e:
                    print(f"Error parsing cluster_sizes: {e}")
                    value = value_str
            else:
                # Try to parse the value as a Python object
                try:
                    # Special handling for dictionaries
                    if value_str.startswith('{') and value_str.endswith('}'):
                        # Use eval with safe globals to parse the dictionary
                        value = eval(value_str, {"__builtins__": {}})
                    else:
                        value = value_str
                except Exception:
                    value = value_str
            
            insights_data[current_section][key] = value
        i += 1
    
    # Post-process for specific sections
    if 'anomaly_detection' in insights_data:
        if 'anomaly_counts' in insights_data['anomaly_detection']:
            anomaly_counts = insights_data['anomaly_detection']['anomaly_counts']
            if isinstance(anomaly_counts, str):
                try:
                    # Parse dictionary format
                    anomaly_counts = anomaly_counts.strip('{}')
                    pairs = anomaly_counts.split(',')
                    value = {}
                    for pair in pairs:
                        if ':' in pair:
                            k, v = pair.split(':')
                            value[k.strip("'")] = int(v.strip())
                    insights_data['anomaly_detection']['anomaly_counts'] = value
                except Exception as e:
                    print(f"Error parsing anomaly_counts: {e}")
    
    return insights_data

def view_insights_visualized():
    """Display insights with visualizations"""
    # Parse insights from the text file
    insights = parse_insights_txt()
    
    # Create the main window
    root = tk.Tk()
    root.title("Bitcoin Price Prediction Insights Visualization")
    root.geometry("1200x800")
    
    # Create a notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create tabs for each section of insights
    for section_name, section_data in insights.items():
        # Create a frame for this section
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text=section_name.upper())
        
        # Create visualizations based on the section type
        if section_name == 'time_series':
            create_time_series_tab(frame, section_data)
        elif section_name == 'regression':
            create_regression_tab(frame, section_data)
        elif section_name == 'classification':
            create_classification_tab(frame, section_data)
        elif section_name == 'clustering':
            create_clustering_tab(frame, section_data)
        elif section_name == 'anomaly_detection':
            create_anomaly_tab(frame, section_data)
    
    root.mainloop()

def create_time_series_tab(frame, time_series_data):
    """Create visualizations for time series insights"""
    # Create a label for the best model
    best_model = time_series_data.get('best_model', 'Unknown')
    ttk.Label(
        frame, 
        text=f"Best Time Series Model: {best_model.upper()}", 
        font=("Arial", 14, "bold")
    ).pack(pady=10)
    
    # Get model performance data
    model_perf = time_series_data.get('model_performance', {})
    
    # Create a figure for the performance comparison
    fig = plt.Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Create a table for the data
    columns = ['Model', 'RMSE', 'MAE', 'MAPE']
    table_data = []
    
    # Prepare bar chart data
    models = []
    rmse_values = []
    mae_values = []
    mape_values = []
    
    for model_name, metrics in model_perf.items():
        models.append(model_name.upper())
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0) 
        mape = metrics.get('mape', 0)
        
        rmse_values.append(rmse)
        mae_values.append(mae)
        mape_values.append(mape)
        
        table_data.append([model_name.upper(), f"{rmse:.2f}", f"{mae:.2f}", f"{mape:.2f}%"])
    
    # Create the bar chart
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, rmse_values, width, label='RMSE')
    ax.bar(x, mae_values, width, label='MAE')
    ax.bar(x + width, [m*100 for m in mape_values], width, label='MAPE (%)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Value')
    ax.set_title('Time Series Model Performance')
    ax.legend()
    
    # Add the figure to the frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Create a frame for the table
    table_frame = ttk.Frame(frame)
    table_frame.pack(fill=tk.X, pady=10)
    
    # Add the table headers
    for i, col in enumerate(columns):
        ttk.Label(
            table_frame, 
            text=col, 
            font=("Arial", 12, "bold"),
            anchor=tk.CENTER,
            width=15
        ).grid(row=0, column=i, sticky=tk.NSEW, padx=2, pady=2)
    
    # Add the table data
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            ttk.Label(
                table_frame, 
                text=cell,
                anchor=tk.CENTER,
                width=15
            ).grid(row=i+1, column=j, sticky=tk.NSEW, padx=2, pady=2)

def create_regression_tab(frame, regression_data):
    """Create visualizations for regression insights"""
    # Create a label for the best model
    best_model = regression_data.get('best_model', 'Unknown')
    ttk.Label(
        frame, 
        text=f"Best Regression Model: {best_model.upper()}", 
        font=("Arial", 14, "bold")
    ).pack(pady=10)
    
    # Get model performance data
    model_perf = regression_data.get('model_performance', {})
    
    # Create figures for the performance comparison
    fig = plt.Figure(figsize=(10, 8))
    
    # RMSE and MAE subplot
    ax1 = fig.add_subplot(211)
    
    # R2 subplot
    ax2 = fig.add_subplot(212)
    
    # Prepare table and chart data
    columns = ['Model', 'RMSE', 'MAE', 'R²', 'MAPE']
    table_data = []
    
    models = []
    rmse_values = []
    mae_values = []
    r2_values = []
    mape_values = []
    
    for model_name, metrics in model_perf.items():
        models.append(model_name.upper())
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        mape = metrics.get('mape', 0)
        
        rmse_values.append(rmse)
        mae_values.append(mae)
        r2_values.append(r2)
        mape_values.append(mape)
        
        table_data.append([
            model_name.upper(), 
            f"{rmse:.2f}", 
            f"{mae:.2f}", 
            f"{r2:.4f}",
            f"{mape:.2f}%"
        ])
    
    # Create the RMSE and MAE bars
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, rmse_values, width, label='RMSE')
    ax1.bar(x + width/2, mae_values, width, label='MAE')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel('Value')
    ax1.set_title('Regression Error Metrics')
    ax1.legend()
    
    # Create the R2 bars
    ax2.bar(x, r2_values, 0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score by Model')
    
    # Add horizontal line at R²=0
    ax2.axhline(y=0, color='r', linestyle='-')
    
    fig.tight_layout()
    
    # Add the figure to the frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Create a frame for the table
    table_frame = ttk.Frame(frame)
    table_frame.pack(fill=tk.X, pady=10)
    
    # Add the table headers
    for i, col in enumerate(columns):
        ttk.Label(
            table_frame, 
            text=col, 
            font=("Arial", 12, "bold"),
            anchor=tk.CENTER,
            width=12
        ).grid(row=0, column=i, sticky=tk.NSEW, padx=2, pady=2)
    
    # Add the table data
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            ttk.Label(
                table_frame, 
                text=cell,
                anchor=tk.CENTER,
                width=12
            ).grid(row=i+1, column=j, sticky=tk.NSEW, padx=2, pady=2)

def create_classification_tab(frame, classification_data):
    """Create visualizations for classification insights"""
    # Create a label for the best model
    best_model = classification_data.get('best_model', 'Unknown')
    ttk.Label(
        frame, 
        text=f"Best Classification Model: {best_model.upper()}", 
        font=("Arial", 14, "bold")
    ).pack(pady=10)
    
    # Get model performance data
    model_perf = classification_data.get('model_performance', {})
    
    # Create a figure for the performance comparison
    fig = plt.Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Create a table for the data
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    table_data = []
    
    # Prepare bar chart data
    models = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for model_name, metrics in model_perf.items():
        models.append(model_name.upper())
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        
        table_data.append([
            model_name.upper(), 
            f"{accuracy:.2f}", 
            f"{precision:.2f}", 
            f"{recall:.2f}",
            f"{f1:.2f}"
        ])
    
    # Create the bar chart
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracy_values, width, label='Accuracy')
    ax.bar(x - 0.5*width, precision_values, width, label='Precision')
    ax.bar(x + 0.5*width, recall_values, width, label='Recall')
    ax.bar(x + 1.5*width, f1_values, width, label='F1')
    
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score')
    ax.set_title('Classification Model Performance')
    ax.legend()
    
    # Add the figure to the frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Create a frame for the table
    table_frame = ttk.Frame(frame)
    table_frame.pack(fill=tk.X, pady=10)
    
    # Add the table headers
    for i, col in enumerate(columns):
        ttk.Label(
            table_frame, 
            text=col, 
            font=("Arial", 12, "bold"),
            anchor=tk.CENTER,
            width=12
        ).grid(row=0, column=i, sticky=tk.NSEW, padx=2, pady=2)
    
    # Add the table data
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            ttk.Label(
                table_frame, 
                text=cell,
                anchor=tk.CENTER,
                width=12
            ).grid(row=i+1, column=j, sticky=tk.NSEW, padx=2, pady=2)

def create_clustering_tab(frame, clustering_data):
    """Create visualizations for clustering insights"""
    # Get cluster profiles
    cluster_profiles = clustering_data.get('cluster_profiles', {})
    cluster_sizes = clustering_data.get('cluster_sizes', {})
    
    if not cluster_sizes:
        ttk.Label(
            frame, 
            text="No clustering data available",
            font=("Arial", 14, "bold")
        ).pack(pady=20)
        return
    
    # Create a figure for the cluster sizes
    sizes_fig = plt.Figure(figsize=(6, 4))
    sizes_ax = sizes_fig.add_subplot(111)
    
    # Convert cluster_sizes to lists
    cluster_labels = [f"Cluster {c}" for c in cluster_sizes.keys()]
    sizes = list(cluster_sizes.values())
    
    # Plot cluster sizes pie chart
    sizes_ax.pie(
        sizes, 
        labels=cluster_labels, 
        autopct='%1.1f%%',
        startangle=90
    )
    sizes_ax.set_title('Cluster Sizes')
    
    # Add the figure to the frame
    sizes_canvas = FigureCanvasTkAgg(sizes_fig, master=frame)
    sizes_canvas.draw()
    sizes_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10, padx=10)
    
    # Create a figure for the cluster profiles
    if cluster_profiles:
        # Get feature names
        features = list(cluster_profiles.keys())
        
        # Create a figure for the cluster profiles
        profiles_fig = plt.Figure(figsize=(8, 6))
        profiles_ax = profiles_fig.add_subplot(111)
        
        # Get cluster numbers
        cluster_nums = set()
        for feature in features:
            if isinstance(cluster_profiles[feature], dict):
                cluster_nums.update(cluster_profiles[feature].keys())
        cluster_nums = sorted(list(cluster_nums))
        
        if cluster_nums and features:
            # Prepare data for visualization
            x = np.arange(len(features))
            width = 0.8 / len(cluster_nums) if cluster_nums else 0.8
            
            for i, cluster in enumerate(cluster_nums):
                values = []
                for feature in features:
                    feature_data = cluster_profiles.get(feature, {})
                    if isinstance(feature_data, dict) and str(cluster) in feature_data:
                        values.append(feature_data[str(cluster)])
                    else:
                        values.append(0)
                
                offset = (i - len(cluster_nums)/2 + 0.5) * width
                profiles_ax.bar(x + offset, values, width, label=f'Cluster {cluster}')
            
            profiles_ax.set_xticks(x)
            profiles_ax.set_xticklabels(features)
            profiles_ax.set_ylabel('Value')
            profiles_ax.set_title('Cluster Profiles by Feature')
            profiles_ax.legend()
            
            # Add the figure to the frame
            profiles_canvas = FigureCanvasTkAgg(profiles_fig, master=frame)
            profiles_canvas.draw()
            profiles_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Display cluster information in a table
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM)
        
        # Create table header
        headers = ['Cluster'] + features
        
        for i, header in enumerate(headers):
            ttk.Label(
                table_frame, 
                text=header, 
                font=("Arial", 12, "bold"),
                anchor=tk.CENTER,
                width=12
            ).grid(row=0, column=i, sticky=tk.NSEW, padx=2, pady=2)
        
        # Get all unique cluster numbers
        all_clusters = set()
        for feature in features:
            if isinstance(cluster_profiles[feature], dict):
                all_clusters.update(cluster_profiles[feature].keys())
        all_clusters = sorted(all_clusters)
        
        # Fill table with data
        for i, cluster in enumerate(all_clusters):
            ttk.Label(
                table_frame, 
                text=f"Cluster {cluster}",
                anchor=tk.CENTER,
                width=12
            ).grid(row=i+1, column=0, sticky=tk.NSEW, padx=2, pady=2)
            
            for j, feature in enumerate(features):
                feature_data = cluster_profiles.get(feature, {})
                if isinstance(feature_data, dict) and str(cluster) in feature_data:
                    value = feature_data[str(cluster)]
                    ttk.Label(
                        table_frame, 
                        text=f"{value:.4f}",
                        anchor=tk.CENTER,
                        width=12
                    ).grid(row=i+1, column=j+1, sticky=tk.NSEW, padx=2, pady=2)
                else:
                    ttk.Label(
                        table_frame, 
                        text="N/A",
                        anchor=tk.CENTER,
                        width=12
                    ).grid(row=i+1, column=j+1, sticky=tk.NSEW, padx=2, pady=2)

def create_anomaly_tab(frame, anomaly_data):
    """Create visualizations for anomaly detection insights"""
    # Get anomaly counts and dates
    anomaly_counts = anomaly_data.get('anomaly_counts', {})
    
    if not anomaly_counts:
        ttk.Label(
            frame, 
            text="No anomaly detection data available",
            font=("Arial", 14, "bold")
        ).pack(pady=20)
        return
    
    # Create a figure for the anomaly counts
    counts_fig = plt.Figure(figsize=(6, 4))
    counts_ax = counts_fig.add_subplot(111)
    
    # Convert anomaly_counts to lists
    if isinstance(anomaly_counts, dict):
        models = list(anomaly_counts.keys())
        counts = list(anomaly_counts.values())
        
        # Plot anomaly counts bar chart
        counts_ax.bar(models, counts)
        counts_ax.set_ylabel('Number of Anomalies')
        counts_ax.set_title('Anomalies Detected by Each Model')
        
        # Add the figure to the frame
        counts_canvas = FigureCanvasTkAgg(counts_fig, master=frame)
        counts_canvas.draw()
        counts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    else:
        ttk.Label(
            frame, 
            text=f"Anomaly counts: {anomaly_counts}",
            font=("Arial", 12)
        ).pack(pady=10)
    
    # Create a frame for the dates table
    dates_frame = ttk.Frame(frame)
    dates_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Get all date keys in the anomaly data
    date_keys = [k for k in anomaly_data.keys() if k.endswith('_dates')]
    
    if date_keys:
        # Find the maximum number of dates
        max_dates = 0
        for key in date_keys:
            dates = anomaly_data.get(key, [])
            if isinstance(dates, list):
                max_dates = max(max_dates, len(dates))
            
        if max_dates > 0:
            # Create a table for the anomaly dates
            columns = ['Model'] + ['Date ' + str(i+1) for i in range(max_dates)]
            
            # Create table header
            for i, col in enumerate(columns):
                ttk.Label(
                    dates_frame, 
                    text=col, 
                    font=("Arial", 12, "bold"),
                    anchor=tk.CENTER,
                    width=20
                ).grid(row=0, column=i, sticky=tk.NSEW, padx=2, pady=2)
            
            # Fill table with data
            for i, key in enumerate(date_keys):
                model_name = key.replace('_dates', '')
                dates = anomaly_data.get(key, [])
                
                ttk.Label(
                    dates_frame, 
                    text=model_name,
                    anchor=tk.CENTER,
                    width=20
                ).grid(row=i+1, column=0, sticky=tk.NSEW, padx=2, pady=2)
                
                if isinstance(dates, list):
                    for j, date in enumerate(dates):
                        ttk.Label(
                            dates_frame, 
                            text=date,
                            anchor=tk.CENTER,
                            width=20
                        ).grid(row=i+1, column=j+1, sticky=tk.NSEW, padx=2, pady=2)
        else:
            ttk.Label(
                dates_frame, 
                text="No anomaly dates available",
                font=("Arial", 12)
            ).pack(pady=10)
    else:
        ttk.Label(
            dates_frame, 
            text="No anomaly dates available",
            font=("Arial", 12)
        ).pack(pady=10)

if __name__ == "__main__":
    # Ask user which view they want
    print("How would you like to view the results?")
    print("1. View figures with matplotlib")
    print("2. View figures with interactive viewer")
    print("3. View insights summary")
    print("4. View insights with graphs and tables")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        view_results_matplotlib()
    elif choice == "2":
        view_results_tkinter()
    elif choice == "3":
        view_insights()
    elif choice == "4":
        view_insights_visualized()
    else:
        print("Invalid choice. Please run the script again and choose a valid option.") 