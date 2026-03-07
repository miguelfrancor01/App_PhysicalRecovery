from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os


def generate_report(session_data):

    data = session_data.get_summary()

    # carpeta donde se guardará el reporte
    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/exercise_report_{timestamp}.pdf"

    c = canvas.Canvas(filename, pagesize=letter)

    width, height = letter

    y = height - 50

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "Reporte de Sesión de Ejercicio")

    y -= 40

    c.setFont("Helvetica", 12)

    c.drawString(50, y, f"Ejercicio: {data['exercise']}")
    y -= 25

    c.drawString(50, y, f"Repeticiones totales: {data['total_reps']}")
    y -= 25

    c.drawString(50, y, f"Repeticiones correctas: {data['correct_reps']}")
    y -= 25

    c.drawString(50, y, f"Repeticiones incorrectas: {data['incorrect_reps']}")
    y -= 25

    c.drawString(50, y, f"Duración de la sesión: {data['duration']} segundos")
    y -= 25

    c.drawString(50, y, f"FPS promedio: {data['avg_fps']}")

    y -= 40

    c.drawString(
        50,
        y,
        f"Generado el: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    c.save()

    print(f"\nReporte guardado en: {filename}\n")