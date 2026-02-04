# 🌍 Air Pollution Classification from Image Data using Deep Learning
This project presents an automated system for Air Quality Index (AQI) classification using environmental images and deep learning models.
It aims to provide a cost-effective alternative to traditional sensor-based air quality monitoring systems.

The system classifies images into six AQI categories by learning visual pollution indicators such as haze density, sky color, and visibility.
# 📌 AQI Classes
* Good
* Moderate
* Unhealthy for Sensitive Groups
* Unhealthy
* Very Unhealthy
* Severe
# 🔹 Dataset
* Used the Air Pollution Image Dataset from India and Nepal.
* Dataset contains real-world environmental images labeled with AQI categories.
* Data split into training, validation, and testing sets.
# 🔹 Models Implemented
* Implemented a Custom CNN as a baseline model using Keras.
* Applied transfer learning using ResNet50 for improved feature extraction.
* Implemented GoogLeNet (Inception v3) for multi-scale feature learning and better generalization.
# 🔹 Model Configuration
* Image sizes: 128×128 (CNN), 224×224 (ResNet50), 299×299 (Inception v3).
* Optimizer: Adam.
* Loss function: Categorical Cross-Entropy.
* Regularization techniques: Dropout, Data Augmentation, Early Stopping.
# 🔹 Performance & Results
- Custom CNN achieved 90% validation accuracy.
- ResNet50 achieved 91% validation accuracy with faster convergence.
- Inception v3 achieved the best validation accuracy of 93%.
- ROC–AUC scores ranged between 0.96 and 0.98 across AQI classes.
# 🔹 Evaluation Metrics
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC and AUC curves
# 🔹 Visual Analysis
- Visualized class distribution and sample predictions.
- Plotted training vs validation accuracy and loss curves.
- Generated confusion matrices and multi-class ROC curves.
# 🔹 Model-wise Results
## Custom CNN (Baseline)
![Air-Pollution-Classification-Using-Deep-Learning](/CNNconfusionmatrix.png)
![Air-Pollution-Classification-Using-Deep-Learning](/CNNgraph.png)
## GoogLeNet (Inception v3) – Best Performing Model
![Air-Pollution-Classification-Using-Deep-Learning](/googlenetconfusionmatrix.png)
![Air-Pollution-Classification-Using-Deep-Learning](/googlenetgraph.png)
# 🔹 Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
# 📈 Performance Comparison
 | Model        | Train Accuracy | Validation Accuracy |
|--------------|----------------|---------------------|
| Custom CNN   | 92.35%         | 90%                 |
| ResNet50    | 96.84%         | 91%                 |
| Inception v3| 95.75%         | 93%              |

# 🔹 How to Run
- Clone the repository.
- Open airpollutionV2.ipynb in Jupyter Notebook or Google Colab.
- Run all cells sequentially after setting dataset paths.
# 🔹 Key Takeaways
- Visual pollution indicators can effectively estimate AQI levels.
- Transfer learning improves accuracy but requires careful fine-tuning.
- Inception v3 generalizes better than baseline CNN and ResNet models.
# 🔹 Future Scope
- Integrate attention mechanisms for improved interpretability.
- Extend the system for real-time pollution monitoring.
- Combine image data with sensor and weather data for multi-modal AQI prediction.

