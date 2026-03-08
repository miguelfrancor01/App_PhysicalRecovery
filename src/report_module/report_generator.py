from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os


def generate_report(session_data, exercise_results, image_path=None):

    data = session_data.get_summary()

    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/reporte_ejercicio_{timestamp}.pdf"

    c = canvas.Canvas(filename, pagesize=letter)

    width, height = letter
    y = height - 50

    # =================================
    # TITULO
    # =================================

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "Reporte de Evaluación de Ejercicio")

    y -= 40

    # =================================
    # INFORMACIÓN DEL PACIENTE
    # =================================

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Información del paciente")

    y -= 25
    c.setFont("Helvetica", 12)

    c.drawString(50, y, "Nombre: Adrian Felipe Vargas Rojas")
    y -= 20

    c.drawString(50, y, "Edad: 26 años")
    y -= 20

    c.drawString(50, y, "Sesión número: 3")
    y -= 20

    c.drawString(
        50,
        y,
        "Padecimiento: Tendinopatía del manguito rotador con limitación funcional en elevación del brazo izquierdo."
    )

    y -= 40

    # =================================
    # RESULTADOS DEL EJERCICIO
    # =================================

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Resultados del ejercicio")

    y -= 25
    c.setFont("Helvetica", 12)

    c.drawString(
        50,
        y,
        f"Repeticiones detectadas: {exercise_results['repetitions_detected']}"
    )

    y -= 30

    # =================================
    # ANGULOS
    # =================================

    c.drawString(50, y, "Ángulos alcanzados por repetición:")

    y -= 20

    for angle in exercise_results["angles"]:
        c.drawString(70, y, f"- {angle:.2f} grados")
        y -= 20

    y -= 10

    # =================================
    # PUNTUACIONES
    # =================================

    c.drawString(50, y, "Puntuación por repetición:")

    y -= 20

    for score in exercise_results["scores"]:
        c.drawString(70, y, f"- {score:.2f}%")
        y -= 20

    y -= 10

    # =================================
    # PUNTUACION FINAL
    # =================================

    c.setFont("Helvetica-Bold", 14)

    c.drawString(
        50,
        y,
        f"Puntuación final del ejercicio: {exercise_results['final_score']:.2f}%"
    )

    y -= 40

    # =================================
    # IMAGEN DEL EJERCICIO
    # =================================

    if image_path and os.path.exists(image_path):

        c.setFont("Helvetica", 12)
        c.drawString(50, y, "Captura del ejercicio analizado:")

        y -= 20

        c.drawImage(
            image_path,
            50,
            y - 250,
            width=400,
            height=250
        )

        y -= 260

    # =================================
    # FECHA DEL REPORTE
    # =================================

    c.setFont("Helvetica", 10)

    c.drawString(
        50,
        y,
        f"Reporte generado el: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    c.save()

    print(f"\nReporte guardado en: {filename}\n")