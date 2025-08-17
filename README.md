# 📘 Miniproyectos de Métodos Numéricos

Repositorio con proyectos pequeños de la asignatura **Métodos Numéricos**, enfocados en la resolución de **ecuaciones diferenciales parciales (EDP)** mediante técnicas y algoritmos numéricos.  
Incluye implementaciones en **Python** y documentación en **LaTeX** sobre distintos métodos de aproximación.

---

## 📂 Contenido del repositorio

- `proyecto1/` → **La Chocolatera** (implementación en Python).  
- `proyecto2/` → **Estabilidad y Orden de los Métodos Numéricos** (implementación en Python).  
- `proyecto3/` → **Relación entre Coordenadas** (documentación en LaTeX).  


---

## ⚙️ Configuración del ambiente virtual

Para ejecutar los proyectos en **Python**, se recomienda usar un ambiente virtual.

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/usuario/miniproyectos-metodos-numericos.git
   cd miniproyectos-metodos-numericos
2. **Ambiente virtual**
Para crear el ambiente virtual, primero abre la terminal (CMD o PowerShell) dentro de la carpeta del proyecto y ejecuta el comando:
python -m venv venv.
Esto generará una carpeta llamada venv donde se guardará el entorno.

Luego, debes activar el ambiente virtual. Si usas CMD, escribe venv\Scripts\activate, y si usas PowerShell, escribe .\venv\Scripts\Activate. Una vez activado, en la terminal aparecerá un prefijo (venv) al inicio de la línea, indicando que estás dentro del entorno virtual.

Con el ambiente ya activo, procede a instalar las librerías necesarias para el proyecto. Si el repositorio incluye un archivo requirements.txt, basta con ejecutar pip install -r requirements.txt. En caso de que no exista, puedes instalar manualmente las librerías principales con el comando pip install numpy matplotlib pillow.

Una vez instaladas, podrás ejecutar los scripts del proyecto normalmente.
