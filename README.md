# Automated Waste Segregation Using IoT and Machine Learning  

## 📌 Overview  
This project implements an IoT-based waste segregation system that classifies and sorts waste in real-time using machine learning. It utilizes a MobileNet-based CNN model with TensorFlow Lite for image classification and an embedded system for automated sorting.  

## 🎥 Project Demonstration  
[![Watch the video](https://img.youtube.com/vi/05wm3ccaHDI/maxresdefault.jpg)](https://youtu.be/05wm3ccaHDI)  
🔗 **[Click here to watch the demo](https://youtu.be/05wm3ccaHDI)**  

## 🚀 Features  
- **Real-time Waste Classification:** Uses MobileNet with TensorFlow Lite for edge-based inference.  
- **Automated Sorting:** Intel Edison controls stepper motors to direct waste into appropriate bins.  
- **User Feedback:** LCD module displays classification results for better interaction.  
- **Efficient & Scalable:** Reduces manual intervention and improves recycling efficiency.  

## 🛠️ Tech Stack  
- **Programming Languages:** Python  
- **Machine Learning:** TensorFlow Lite, MobileNet, Google Teachable Machine  
- **Hardware:** Intel Edison, Stepper Motors, LCD RGB Backlight  
- **Embedded Systems:** IoT, Edge Computing  

## 📦 System Architecture  
1. A **camera** captures images of waste.  
2. A **TensorFlow Lite model** classifies the item as recyclable or non-recyclable.  
3. **Intel Edison** processes the classification and controls the **stepper motor** to direct waste.  
4. The **LCD module** displays the classification result in real-time.  

## 🔗 Important Links  
### **Main IoT Kit & Board Resources:**  
- [Grove Starter Kit Plus – IoT Edition](https://seeeddoc.github.io/Grove_Starter_Kit_Plus-IoT_Edition/)  
- [Grove Starter Kit – Getting Started Guide](https://ssg-drd-iot.github.io/getting-started-guides/docs/sensor_examples/grove_starter_kit/details-green_box.html)  

### **Stepper Motor Driver Resources:**  
- [Gear Stepper Motor Driver Pack](https://wiki.seeedstudio.com/Gear_Stepper_Motor_Driver_Pack/)  
