# 🧬 Fibrosis Detection in Histopathological Images (MATLAB + U-Net)

## 📖 Descriere
Acest proiect implementează un **sistem de detecție și segmentare a fibrozei** în imagini histopatologice, utilizând un model **U-Net** antrenat în MATLAB.  
Scopul este identificarea automată a zonelor afectate de fibroză și cuantificarea procentului de țesut afectat.  

---

## 🗂 Structură directoare
- **📁 split_1/train/images** – imagini de antrenare  
- **📁 split_1/train/masks_rgb** – măști de antrenare (zone de fibroză = verde)  
- **📁 split_1/val/images** – imagini de validare  
- **📁 split_1/val/masks_rgb** – măști de validare  
- **📁 models/** – modele U-Net salvate după antrenare  

---

## ⚙️ Funcționalități
✔️ Antrenare U-Net cu augmentare de date (rotații, scalări, flip-uri)  
✔️ Detectarea automată a zonelor de fibroză în imagini histopatologice  
✔️ Calcul procentual al suprafeței afectate  
✔️ Evaluare globală cu metrici numerice și matrice de confuzie  
✔️ Vizualizare grafică a performanței (ROC Curve, PR Curve)  

---

## 📊 Metrici calculate
- **Dice Coefficient**  
- **Intersection over Union (IoU)**  
- **Precision, Recall, F1-Score**  
- **Accuracy**  
- **ROC Curve și AUC**  
- **Precision-Recall Curve**  

---

## 🚀 Cum rulezi proiectul
1. Clonează repository-ul și pregătește folderele `train` și `val` cu imaginile și măștile.  
2. Rulează scriptul de **antrenare** pentru a genera modelul U-Net:  
   ```matlab
   train_unet.m
3.Rulează scriptul de evaluare și testare pentru rezultate și vizualizări:
test_unet.m
4.Rezultatele vor fi afișate în MATLAB: metrici, matrice de confuzie, ROC și PR curve, plus imagini cu overlay al fibrozei.
