from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="NMPDE Academic Dashboard")
app.mount("/results", StaticFiles(directory="../results"), name="results")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# --- NEW METHODOLOGY ROUTES ---
@app.get("/leapfrog", response_class=HTMLResponse)
async def leapfrog_page(request: Request):
    return templates.TemplateResponse(request=request, name="leapfrog.html")

@app.get("/rk4", response_class=HTMLResponse)
async def rk4_page(request: Request):
    return templates.TemplateResponse(request=request, name="rk4.html")

@app.get("/sparse", response_class=HTMLResponse)
async def sparse_page(request: Request):
    return templates.TemplateResponse(request=request, name="sparse.html")

@app.get("/scaling", response_class=HTMLResponse)
async def scaling_page(request: Request):
    return templates.TemplateResponse(request=request, name="scaling.html")

@app.get("/visualization", response_class=HTMLResponse)
async def visualization(request: Request):
    return templates.TemplateResponse(request=request, name="visualization.html")