import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.agent import MedicalAgent
from src.graph.graph_loader import MedicalGraphLoader
from src.graph.graph_reasoner import MedicalGraphReasoner
from src.decision.decision_engine import ClinicalDecisionEngine
from src.api.predict_router import router as predict_router
from src.api.agent_router import router as agent_router
from src.api.workflow_router import router as workflow_router
from src.api.health_router import router as health_router
from src.api.wearable_router import router as wearable_router
from src.api.xai_router import router as xai_router
from src.api.fhir_router import router as fhir_router
from src.api.eval_router import router as eval_router
from src.api.treatment_router import router as treatment_router
from src.api.vision_router import router as vision_router
from src.api.privacy_router import router as privacy_router
from src.api.multimodal_router import router as multimodal_router
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        logger.info("Starting up FastAPI application...")
        logger.info("Initializing Medical NER Predictor model...")
        start_time = time.time()
        from src.inference.predict_ner import MedicalNERPredictor

        predictor = MedicalNERPredictor()
        app_instance.state.predictor = predictor
        app_instance.state.medical_agent = MedicalAgent(ner_predictor=predictor)

        logger.info("Loading and building Medical Knowledge Graph...")
        graph = MedicalGraphLoader().load_or_build()
        app_instance.state.graph_reasoner = MedicalGraphReasoner(graph)
        app_instance.state.decision_engine = ClinicalDecisionEngine(graph)

        logger.info("Model, agent, graph, and decision engine loaded in %.2f seconds.", time.time() - start_time)
        yield

    app = FastAPI(
        title="MedFusion-CXR — Multimodal Multi-Agent Clinical Decision Support System",
        description="Integrating Pretrained DenseNet-121 (NIH ChestX-ray14), XGBoost + TreeSHAP, and Gated Feature Fusion within a Multi-Agent Clinical Architecture.",
        version="2.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(multimodal_router)
    app.include_router(treatment_router)
    app.include_router(predict_router)
    app.include_router(agent_router)
    app.include_router(workflow_router)
    app.include_router(health_router)
    app.include_router(wearable_router)
    app.include_router(xai_router)
    app.include_router(fhir_router)
    app.include_router(eval_router)
    app.include_router(vision_router)
    app.include_router(privacy_router)

    # Mount Web Frontend UI
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
    if os.path.exists(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")

    @app.get("/", include_in_schema=False)
    async def root():
        """Automatically redirects root http://localhost:8000 to full styled Web Dashboard UI"""
        return RedirectResponse(url="/web/index.html")

    return app
