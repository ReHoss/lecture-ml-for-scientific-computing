# APM52009: Machine Learning for Scientific Computing and Numerical Analysis

Welcome to the official repository for the **Machine Learning for Scientific Computing and Numerical Analysis (APM52009)** course, taught in the 2026 academic year.

**Instructors:** Prof. Victorita Dolean & Prof. Hadrien Montanelli  
**Teaching Team:** The APM52009 Academic Staff  

---

## 📖 About the Course

How is modern machine learning revolutionizing traditional computational sciences? This course bridges the gap between classical numerical analysis and the rapidly evolving field of **Scientific Machine Learning (SciML)**. 

By blending deep learning architectures with rigorous, physics-based modeling, SciML provides powerful new tools for solving complex physical phenomena, simulating partial differential equations (PDEs), and addressing large-scale optimization challenges. 

Throughout this course, students will explore the theoretical foundations and hands-on applications of Neural Networks in scientific computing. Whether delving into the mathematical rigors of approximation theory, tackling classic numerical methods, or implementing cutting-edge Operator Learning and Physics-Informed Neural Networks (PINNs) in PyTorch, this curriculum offers a deep-dive into the methods shaping the future of scientific research and engineering.

---

## 📚 Course Syllabus & Lecture Notes

The curriculum is carefully structured to map robust theory to impactful practical applications:

*   **Week 1:** Introduction & Scope of Scientific Machine Learning
*   **Week 2:** Neural Network Approximation Theory
*   **Week 3:** Optimization
*   **Week 4:** Introduction to PDEs and Classical Numerical Methods
*   **Week 5:** Physics-Informed Neural Networks (PINNs)
*   **Week 6:** Introduction to Parametric PDEs and Reduced Basis Methods
*   **Week 7:** Operator Learning
*   **Week 8:** Randomized Numerical Linear Algebra in SciML

> 📌 **Lecture Notes:** The complete reading material is available in the [`lecture_notes/`](./lecture_notes) directory as a single merged resource (`ml_for_sci_computing_lecture_notes_montanelli_dolean_2026.pdf`), alongside individual week-by-week chapter files.

---

## 💻 Practical Sessions (PCs)

Theory is only as good as its implementation. The [`practical_sessions/`](./practical_sessions) directory contains the Jupyter Notebooks (provided without solutions) and necessary helper scripts. These sessions are designed to give you continuous, hands-on experience with the concepts discussed in lectures.

### Environment Setup

To run the practical sessions, you will need a fully equipped Python environment containing libraries such as PyTorch, NumPy, SciPy, and Matplotlib. We strongly recommend using `venv` or `conda` to manage your dependencies cleanly:

```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-folder>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

---
*We look forward to an exciting semester exploring the frontiers of scientific computing with you!*
