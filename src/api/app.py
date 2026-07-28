import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clinical Intelligence Platform API",
        description="Modular FastAPI service for Clinical NLP, Knowledge Graph, CDSS, Wearable AI, XAI, FHIR, and Evaluation.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predict_router)
    app.include_router(agent_router)
    app.include_router(workflow_router)
    app.include_router(health_router)
    app.include_router(wearable_router)
    app.include_router(xai_router)
    app.include_router(fhir_router)
    app.include_router(eval_router)

    predictor: Any | None = None
    medical_agent: MedicalAgent | None = None
    graph_reasoner: MedicalGraphReasoner | None = None
    clinical_decision_engine: ClinicalDecisionEngine | None = None

    @app.on_event("startup")
    def startup_event() -> None:
        nonlocal predictor, medical_agent, graph_reasoner, clinical_decision_engine
        logger.info("Starting up FastAPI application...")
        logger.info("Initializing Medical NER Predictor model...")
        start_time = time.time()
        from src.inference.predict_ner import MedicalNERPredictor

        predictor = MedicalNERPredictor()
        medical_agent = MedicalAgent(ner_predictor=predictor)
        app.state.predictor = predictor
        
        logger.info("Loading and building Medical Knowledge Graph...")
        graph_loader = MedicalGraphLoader()
        graph = graph_loader.load_or_build()
        graph_reasoner = MedicalGraphReasoner(graph)
        clinical_decision_engine = ClinicalDecisionEngine(graph)
        app.state.decision_engine = clinical_decision_engine
        
        logger.info("Model, agent, graph, and decision engine loaded in %.2f seconds.", time.time() - start_time)

    return app
