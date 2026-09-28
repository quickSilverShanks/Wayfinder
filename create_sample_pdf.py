import os
from pathlib import Path


def generate_sample_pdf_with_reportlab(file_path: Path, title: str, content: str):
    """Generates a PDF using reportlab if installed, or falls back to text PDF format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(str(file_path), pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, title)
        c.setFont("Helvetica", 10)
        
        y = 720
        for line in content.split("\n"):
            if y < 72:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = 750
            c.drawString(72, y, line[:90])
            y -= 15
            
        c.save()
        print(f"Generated PDF with ReportLab: {file_path}")
        return
    except ImportError:
        pass

    # Basic PDF raw stream creator if reportlab is not installed
    pdf_content = (
        f"%PDF-1.4\n"
        f"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        f"2 0 obj <</Type /Pages /Kinds [3 0 R] /Count 1>> endobj\n"
        f"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R>> endobj\n"
        f"4 0 obj <</Length 200>> stream\n"
        f"BT /F1 12 Tf 72 712 Td ({title}) Tj ET\n"
        f"BT /F1 10 Tf 72 680 Td ({content[:100]}) Tj ET\n"
        f"endstream endobj\n"
        f"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000115 00000 n \n0000000215 00000 n \n"
        f"trailer <</Size 5 /Root 1 0 R>>\nstartxref\n450\n%%EOF"
    )
    with open(file_path, "wb") as f:
        f.write(pdf_content.encode("latin1"))
    print(f"Generated raw PDF structure: {file_path}")


def create_all_sample_documents(base_dir: str = "./data/source_documents"):
    base_path = Path(base_dir)

    # Document 1: HR Policy (Category: HR, SubCategory: Policies)
    hr_file = base_path / "HR" / "Policies" / "leave_policy.pdf"
    hr_title = "Wayfinder Enterprise Annual Leave Policy"
    hr_body = (
        "1. Executive Overview\n"
        "Frontline staff members are entitled to 20 business days of paid annual leave per calendar year.\n"
        "Requests must be submitted 2 weeks in advance via the Wayfinder portal.\n\n"
        "2. Emergency Leave & Support Contact\n"
        "For urgent leave inquiries, contact HR support representative John Doe at john.doe@wayfinder.internal "
        "or call direct extension +1 (555) 019-2834.\n"
        "Employee SSN reference for identity verification is 987-65-4321.\n\n"
        "3. Rollover Rules\n"
        "A maximum of 5 unused leave days can be rolled over to the subsequent calendar year."
    )
    generate_sample_pdf_with_reportlab(hr_file, hr_title, hr_body)

    # Document 2: IT Security (Category: IT, SubCategory: Security)
    it_file = base_path / "IT" / "Security" / "password_guidelines.pdf"
    it_title = "Wayfinder IT Security & Authentication Standards"
    it_body = (
        "1. Password Requirements\n"
        "All employee accounts must use multi-factor authentication (MFA) and passwords of at least 16 characters.\n"
        "Passwords must be updated every 90 days.\n\n"
        "2. Helpdesk Contact & Escalation\n"
        "If you suspect a credential leak, notify IT security immediately at security@wayfinder.internal or call +1-800-555-0199.\n"
        "Internal security server IPv4 address for logs is 192.168.1.105.\n"
        "Corporate admin card reference: 4532-1234-5678-9010."
    )
    generate_sample_pdf_with_reportlab(it_file, it_title, it_body)

    print("\nSample PDF documents created successfully under", base_dir)


if __name__ == "__main__":
    create_all_sample_documents()
