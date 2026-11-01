# Plan d'article — à adapter au template ESAIP

**Titre de travail :** « Réexamen méthodologique d'une comparaison CNN–Vision Transformer pour la classification d'images aériennes civiles »

Ce plan est une aide de rédaction, **pas** un remplacement du template officiel obligatoire à télécharger sur Moodle.

## Résumé / Abstract (anglais)
Contexte de la robustesse de la perception visuelle, question de recherche, corpus historique, protocole évalué, limites observées et principales conclusions vérifiées. Ne pas annoncer une amélioration non mesurée.

## 1. Introduction
Évolution du PSIR1 (hypothèses architecture / robustesse / explicabilité / calcul) vers le protocole PSIR2 (ResNet50, DeiT-Tiny, VisDrone). Expliquer la portée volontairement limitée aux observations civiles.

## 2. État de l'art actualisé
Classificateurs CNN et ViT ; problèmes de robustesse adversariale et de transfert d'apprentissage ; qualité méthodologique des jeux de données ; évaluation et confiance des classificateurs ; limites de transférabilité. Vérifier individuellement les articles cités et compléter jusqu'aux travaux disponibles en 2026.

## 3. Matériels et méthodes
Historique du recadrage VisDrone et des classes ; provenance des images et partitions ; modèles et préentraînement ; attaques historiques white-box FGSM / PGD ; détails de reproductibilité ; métriques descriptives et limites de l'échantillon.

## 4. Résultats
Séparer soigneusement :
- **Historique PSIR2 :** chiffres et figures tels que rapportés l'an dernier, sans validation indépendante.
- **Audit PARE1 :** observations démontrables par inspection du code.
- **Résultats nouveaux :** uniquement les sorties d'expériences réellement exécutées et archivées.

## 5. Discussion
Pourquoi échantillon de 20 images, découpages par recadrage, absence de domaines indépendants et comparaison non hybride limitent les conclusions. Différence entre confiance, calibration et contrôle humain. Les données civiles ne démontrent pas la fiabilité d'applications militaires.

## 6. Conclusion
Bilan transparent des hypothèses **confirmées / infirmées / non évaluées**. Perspectives d'évaluation méthodologique et état de l'art. Éviter les formulations de garantie d'invulnérabilité ou d'embarquabilité.

## Références, contributions et reproductibilité
Bibliographie conforme à la trame ; DOI/arXiv vérifiés ; contributions des quatre membres ; référence à la version Git exacte, aux données et aux conditions d'exécution.
