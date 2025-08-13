# Safety & Ethical Considerations

This document outlines the critical safety and ethical guidelines governing this project.

## 1. Not a Medical Device

**CRITICAL**: This system is a **research and demonstration tool only**.

-   It is **NOT** a certified medical device and has not undergone regulatory approval (e.g., from the FDA or EMA).
-   It must **NEVER** be used for primary clinical diagnosis, making treatment decisions, or any form of real-world patient care.
-   The outputs are probabilistic and illustrative; they are not a substitute for the judgment of a qualified medical professional.

All user interfaces (e.g., the Streamlit demo) must display a prominent disclaimer stating this.

## 2. Language and Communication

To prevent misinterpretation, the system's output must avoid definitive or prescriptive language.

-   **Use:** "Suggestive of...", "Consistent with patterns of...", "AI confidence...", "Recommend follow-up by a specialist."
-   **Avoid:** "Patient has [Disease Name]", "Diagnosis is...", "This is [Disease Name]".

The goal is to provide a supportive second opinion, not a definitive answer.

## 3. Data Privacy and Anonymization

All patient data is highly sensitive and protected by regulations like HIPAA.

-   **Anonymization**: Before any image is processed, all Protected Health Information (PHI) must be stripped. This includes patient names, IDs, dates, and any other identifying text burned into the image. Our `data_prep` pipeline includes a placeholder for hashing filenames, but a real-world system would require robust DICOM de-identification tools.
-   **Data Security**: The system is designed for on-premise or private cloud deployment. No patient data should be sent to third-party APIs without explicit, secure, and compliant agreements.
-   **Sample Data**: The data included in this repository is from public, anonymized datasets (or is synthetically generated) and contains no real PHI.