#  Machine Learning for Scientific Computing and Numerical Analysis

Welcome to the official repository for the **Machine Learning for Scientific Computing and Numerical Analysis** course, taught in the 2026 academic year at [École Polytechnique](https://www.polytechnique.edu/) and the [Eindhoven University of Technology](https://www.tue.nl/en/) within the [EuroteQ program](https://euroteq.eurotech-universities.eu/). 

**Principal Lecturer:** [Hadrien Montanelli](https://hadrien-montanelli.github.io/) & [Victorita Dolean](https://www.tue.nl/en/research/researchers/victorita-dolean-maini)  
**Teaching Assistant:**  [Rémy Hosseinkhan-Boucher](https://rehoss.github.io/)

---

## About the Course

How is modern machine learning revolutionizing traditional computational sciences? This course bridges the gap between classical numerical analysis and the rapidly evolving field of **Scientific Machine Learning (SciML)**. 

By blending deep learning architectures with rigorous, physics-based modeling, SciML provides powerful new tools for solving complex physical phenomena, simulating partial differential equations (PDEs), and addressing large-scale optimization challenges. 

Throughout this course, students will explore the theoretical foundations and hands-on applications of learning algorithms in scientific computing. Whether delving into the mathematical rigors of approximation theory, tackling classic numerical methods, or implementing cutting-edge Operator Learning and Physics-Informed Neural Networks (PINNs), this curriculum offers a deep-dive into the methods shaping the future of scientific research and engineering.

### Prerequisites

To get the most out of this course, students are expected to have a solid foundation in:
- **Mathematics:** Multivariable calculus, linear algebra, and basic probability/statistics. Familiarity with ordinary and partial differential equations (ODEs/PDEs) is highly recommended.
- **Programming:** Basic proficiency in Python, as the practical sessions heavily rely on libraries such as NumPy, SciPy, and PyTorch.

---

## Course Syllabus & Lecture Notes

The curriculum is carefully structured to map robust theory to practical applications:

*   **Week 1:** Introduction & Scope of Scientific Machine Learning
*   **Week 2:** Neural Network Approximation Theory
*   **Week 3:** Optimization
*   **Week 4:** Introduction to PDEs and Classical Numerical Methods
*   **Week 5:** Physics-Informed Neural Networks (PINNs)
*   **Week 6:** Introduction to Parametric PDEs and Reduced Basis Methods
*   **Week 7:** Operator Learning
*   **Week 8:** Randomized Numerical Linear Algebra in SciML

> 📌 **Lecture Notes:** The complete reading material for this course is openly available on **HAL (Hyper Article en Ligne)**. 
> *[Link to the HAL archive will be provided here once published]*

---

## Practical Sessions

The [`practical_sessions/`](./practical_sessions) directory contains the Jupyter Notebooks (provided without solutions) and necessary helper scripts. These sessions are designed to give you continuous, hands-on experience with the concepts discussed in lectures.

### Environment Setup

> 💡 **Hardware Requirements:** All practical sessions are designed to run efficiently on a standard laptop CPU; **no GPU is required**.

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
