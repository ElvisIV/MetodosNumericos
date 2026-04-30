from flask import Flask, render_template, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# ─────────────────────────────────────────────
# SYSTEM DEFINITIONS
# ─────────────────────────────────────────────

SYSTEMS = {
    "ideal": {
        "A": [[20, 2, 1], [3, 15, 4], [1, 2, 10]],
        "b": [100, 120, 80],
        "name": "Caso Ideal",
        "subtitle": "Sistema Eficiente (Tráfico Normal)",
        "description": "Los servicios están bien diferenciados. Auth consume CPU, Catálogo RAM y Pagos Ancho de Banda. La matriz es diagonal dominante.",
        "badge": "ESTABLE",
        "badge_class": "badge-ideal"
    },
    "stress": {
        "A": [[800, 150, 100], [120, 900, 200], [100, 150, 750]],
        "b": [10000, 15000, 9000],
        "name": "Caso Bajo Estrés",
        "subtitle": "Black Friday / Alta Demanda",
        "description": "Demanda masiva. Todos los servicios al máximo. SOR y Gradiente Conjugado demuestran su superioridad en convergencia.",
        "badge": "ALTO ESTRÉS",
        "badge_class": "badge-stress"
    },
    "ill": {
        "A": [[10, 5, 2], [10, 5.0001, 2], [1, 1, 8]],
        "b": [50, 50.0001, 40],
        "name": "Caso Mal Condicionado",
        "subtitle": "Redundancia Crítica",
        "description": "Auth y Catálogo configurados para tareas casi idénticas. Hiperplanos casi paralelos. Convergencia extremadamente difícil.",
        "badge": "INESTABLE",
        "badge_class": "badge-ill"
    }
}

# ─────────────────────────────────────────────
# NUMERICAL METHODS
# ─────────────────────────────────────────────

def solve_exact(A, b):
    """Exact solution using numpy"""
    try:
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)
        x = np.linalg.solve(A_np, b_np)
        return x.tolist(), None
    except np.linalg.LinAlgError as e:
        return None, str(e)

def lu_factorization(A, b):
    """LU Factorization (Doolittle) - Direct Method"""
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    steps = []

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - s

        for j in range(i+1, n):
            if abs(U[i][i]) < 1e-14:
                return None, None, None, float('inf'), "División por cero en LU", []
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - s) / U[i][i]

    # Forward substitution Ly = b
    y = [0.0]*n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j]*y[j] for j in range(i))

    # Backward substitution Ux = y
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        if abs(U[i][i]) < 1e-14:
            return None, None, None, float('inf'), "División por cero en sustitución", []
        x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]

    residual = np.linalg.norm(np.dot(A, x) - np.array(b))
    return x, L, U, residual, None, steps

def jacobi(A, b, tol=1e-6, max_iter=1000):
    """Jacobi iterative method"""
    n = len(A)
    x = [0.0]*n
    history = []
    errors = []

    for it in range(max_iter):
        x_new = [0.0]*n
        for i in range(n):
            if abs(A[i][i]) < 1e-14:
                return None, it, errors, False, "Diagonal cero detectada"
            s = sum(A[i][j]*x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        err = max(abs(x_new[i] - x[i]) for i in range(n))
        errors.append(err)
        history.append(x_new[:])
        x = x_new[:]

        if err < tol:
            return x, it+1, errors, True, None

    return x, max_iter, errors, False, "No convergió"

def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
    """Gauss-Seidel iterative method"""
    n = len(A)
    x = [0.0]*n
    errors = []

    for it in range(max_iter):
        x_old = x[:]
        for i in range(n):
            if abs(A[i][i]) < 1e-14:
                return None, it, errors, False, "Diagonal cero"
            s1 = sum(A[i][j]*x[j] for j in range(i))
            s2 = sum(A[i][j]*x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]

        err = max(abs(x[i] - x_old[i]) for i in range(n))
        errors.append(err)

        if err < tol:
            return x, it+1, errors, True, None

    return x, max_iter, errors, False, "No convergió"

def sor(A, b, omega=1.25, tol=1e-6, max_iter=1000):
    """Successive Over-Relaxation (SOR)"""
    n = len(A)
    x = [0.0]*n
    errors = []

    for it in range(max_iter):
        x_old = x[:]
        for i in range(n):
            if abs(A[i][i]) < 1e-14:
                return None, it, errors, False, omega, "Diagonal cero"
            s1 = sum(A[i][j]*x[j] for j in range(i))
            s2 = sum(A[i][j]*x_old[j] for j in range(i+1, n))
            x_gs = (b[i] - s1 - s2) / A[i][i]
            x[i] = (1 - omega)*x_old[i] + omega*x_gs

        err = max(abs(x[i] - x_old[i]) for i in range(n))
        errors.append(err)

        if err < tol:
            return x, it+1, errors, True, omega, None

    return x, max_iter, errors, False, omega, "No convergió"

def preconditioned_conjugate_gradient(A, b, tol=1e-6, max_iter=1000):
    """Preconditioned Conjugate Gradient with Jacobi preconditioner"""
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    n = len(b)

    # Jacobi preconditioner: M = diag(A)
    M_inv = np.diag(1.0 / np.diag(A_np))

    x = np.zeros(n)
    r = b_np - A_np @ x
    z = M_inv @ r
    p = z.copy()
    rz = np.dot(r, z)
    errors = []

    for it in range(max_iter):
        Ap = A_np @ p
        alpha = rz / (np.dot(p, Ap) + 1e-14)
        x = x + alpha * p
        r = r - alpha * Ap

        err = np.linalg.norm(r)
        errors.append(float(err))

        if err < tol:
            return x.tolist(), it+1, errors, True, None

        z = M_inv @ r
        rz_new = np.dot(r, z)
        beta = rz_new / (rz + 1e-14)
        p = z + beta * p
        rz = rz_new

    return x.tolist(), max_iter, errors, False, "No convergió"

def compute_condition_number(A):
    try:
        return float(np.linalg.cond(np.array(A, dtype=float)))
    except:
        return float('inf')

def run_all_methods(A, b):
    """Run all methods on a system and return consolidated results"""
    results = {}

    # Exact
    x_exact, err = solve_exact(A, b)
    results['exact'] = {'x': x_exact, 'error': err}

    # LU
    x_lu, L, U, res, err_lu, _ = lu_factorization(A, b)
    results['lu'] = {
        'x': x_lu,
        'L': L,
        'U': U,
        'residual': float(res) if res != float('inf') else 'inf',
        'error': err_lu,
        'iterations': 1,
        'converged': err_lu is None
    }

    # Jacobi
    x_j, it_j, err_j, conv_j, msg_j = jacobi(A, b)
    results['jacobi'] = {
        'x': x_j,
        'iterations': it_j,
        'errors': err_j,
        'converged': conv_j,
        'message': msg_j
    }

    # Gauss-Seidel
    x_gs, it_gs, err_gs, conv_gs, msg_gs = gauss_seidel(A, b)
    results['gauss_seidel'] = {
        'x': x_gs,
        'iterations': it_gs,
        'errors': err_gs,
        'converged': conv_gs,
        'message': msg_gs
    }

    # SOR
    x_sor, it_sor, err_sor, conv_sor, omega, msg_sor = sor(A, b)
    results['sor'] = {
        'x': x_sor,
        'iterations': it_sor,
        'errors': err_sor,
        'converged': conv_sor,
        'omega': omega,
        'message': msg_sor
    }

    # PCG
    x_pcg, it_pcg, err_pcg, conv_pcg, msg_pcg = preconditioned_conjugate_gradient(A, b)
    results['pcg'] = {
        'x': x_pcg,
        'iterations': it_pcg,
        'errors': err_pcg,
        'converged': conv_pcg,
        'message': msg_pcg
    }

    # Condition number
    results['condition_number'] = compute_condition_number(A)

    return results

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', systems=SYSTEMS)

@app.route('/api/solve/<case_id>')
def solve_case(case_id):
    if case_id not in SYSTEMS:
        return jsonify({'error': 'Caso no encontrado'}), 404

    sys = SYSTEMS[case_id]
    results = run_all_methods(sys['A'], sys['b'])
    return jsonify(results)

@app.route('/api/solve_custom', methods=['POST'])
def solve_custom():
    data = request.json
    try:
        A = data['A']
        b = data['b']
        if len(A) != 3 or any(len(row) != 3 for row in A) or len(b) != 3:
            return jsonify({'error': 'Se requiere sistema 3x3'}), 400
        results = run_all_methods(A, b)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/convergence_history', methods=['POST'])
def convergence_history():
    data = request.json
    A = data['A']
    b = data['b']
    method = data.get('method', 'jacobi')
    tol = float(data.get('tol', 1e-6))

    if method == 'jacobi':
        _, _, errors, conv, msg = jacobi(A, b, tol=tol)
    elif method == 'gauss_seidel':
        _, _, errors, conv, msg = gauss_seidel(A, b, tol=tol)
    elif method == 'sor':
        omega = float(data.get('omega', 1.25))
        _, _, errors, conv, _, msg = sor(A, b, omega=omega, tol=tol)
    elif method == 'pcg':
        _, _, errors, conv, msg = preconditioned_conjugate_gradient(A, b, tol=tol)
    else:
        return jsonify({'error': 'Método desconocido'}), 400

    return jsonify({'errors': errors, 'converged': conv, 'message': msg})

@app.route('/api/plane_data/<case_id>')
def plane_data(case_id):
    if case_id not in SYSTEMS:
        return jsonify({'error': 'Caso no encontrado'}), 404
    sys = SYSTEMS[case_id]
    A = sys['A']
    b = sys['b']
    x_exact, _ = solve_exact(A, b)
    return jsonify({'A': A, 'b': b, 'solution': x_exact})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
