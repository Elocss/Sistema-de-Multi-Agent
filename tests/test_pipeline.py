import os
import unittest
from src.config import GEMINI_API_KEY
from src.vector_store import initialize_vector_stores
from src.pipeline import SupportPipeline

class TestMultiAgentPipeline(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        """Inicializar bases de datos antes de las pruebas."""
        if not GEMINI_API_KEY:
            raise unittest.SkipTest("GEMINI_API_KEY no está configurada. Saltando pruebas integrales.")
        
        # Inicializar Chroma local con los documentos de ejemplo
        initialize_vector_stores()
        cls.pipeline = SupportPipeline()

    async def test_hr_routing_and_rag(self):
        """Verifica que consultas de vacaciones vayan a HR y den la respuesta correcta usando RAG."""
        query = "Hola, me gustaría saber cuántos días de vacaciones tengo acumulados al año."
        result = await self.pipeline.run(query)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_department"], "HR")
        self.assertTrue(result["routing_confidence"] > 0.7)
        # Verificar que la respuesta contenga datos fundamentados en el RAG (15 días hábiles)
        self.assertIn("15", result["response"].lower())
        self.assertIn("días", result["response"].lower())

    async def test_it_routing_and_rag(self):
        """Verifica que consultas de VPN vayan a IT Support y respondan correctamente con RAG."""
        query = "cómo me conecto a la vpn corporativa? no tengo la dirección del servidor."
        result = await self.pipeline.run(query)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_department"], "IT Support")
        # Verificar que mencione el servidor o el cliente Cisco de la guía
        self.assertIn("vpn.empresa.com", result["response"].lower())
        self.assertIn("anyconnect", result["response"].lower())

    async def test_finance_routing_and_rag(self):
        """Verifica que consultas de reembolsos vayan a Finance y respondan correctamente con RAG."""
        query = "Hola, cuál es el monto máximo reembolsable para comida diaria en un viaje corporativo?"
        result = await self.pipeline.run(query)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_department"], "Finance")
        # Verificar que la respuesta contenga el límite de $50 USD
        self.assertIn("50", result["response"].lower())

    async def test_legal_routing_and_rag(self):
        """Verifica que consultas de NDAs vayan a Legal y respondan correctamente con RAG."""
        query = "Necesito firmar un NDA con un nuevo cliente, cuál es el proceso?"
        result = await self.pipeline.run(query)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_department"], "Legal")
        # Verificar que la respuesta mencione LegalHub o NDA Mutuo
        self.assertTrue(
            "legalhub" in result["response"].lower() or 
            "nda" in result["response"].lower() or 
            "mutuo" in result["response"].lower()
        )

if __name__ == "__main__":
    unittest.main()
