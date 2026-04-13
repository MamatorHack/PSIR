@echo off
echo Démarrage du Pipeline d'Expérimentation PSIR...
echo Phase 0 : Data Engineering (VisDrone Proxy SAA)...
python 00_prepare_proxy_dataset.py
if %ERRORLEVEL% neq 0 (
    echo [Erreur] Échec de la préparation VisDrone Proxy.
    exit /b %ERRORLEVEL%
)

echo.
echo Phase 1 : Entraînement Baseline Proxy...
python 01_train.py
if %ERRORLEVEL% neq 0 (
    echo [Erreur] L'entraînement a échoué.
    exit /b %ERRORLEVEL%
)

echo.
echo Phase 2 et 3 : Génération de la Menace et Évaluation de Résilience...
python 02_evaluate.py
if %ERRORLEVEL% neq 0 (
    echo [Erreur] L'évaluation a échoué.
    exit /b %ERRORLEVEL%
)

echo Phase 4 : Évaluation de Transférabilité SAA (Cross-Domain)...
python 03_cross_domain_eval.py
if %ERRORLEVEL% neq 0 (
    echo [Erreur] L'évaluation cross-domain a échoué.
    exit /b %ERRORLEVEL%
)

echo.
echo Phase 5 et 6 : Explainability (XAI) et Calibration Softmax...
python 04_explainability_confidence.py
if %ERRORLEVEL% neq 0 (
    echo [Erreur] La modélisation XAI a échoué.
    exit /b %ERRORLEVEL%
)

echo.
echo ===============================================
echo EXÉCUTION TERMINÉE AVEC SUCCÈS !
echo Les graphiques et matrices ont été générés.
echo ===============================================
pause
