# PARE1 — Révision scientifique du PSIR (2026–2027)

Cette branche conserve **intégralement les scripts historiques PSIR2 sur main** et ajoute une documentation critique et une analyse statistique indépendante des prédictions déjà obtenues. L'objet est de préparer un article PARE1 fidèle à PSIR1 et PSIR2, sans effacer leurs limites.

## Historique à conserver

- **PSIR1 (S7)** : étude de l'IA et des armes létales autonomes, formulation des hypothèses de robustesse, explicabilité, modèles CNN / Transformers ou ensembles et contraintes embarquées.
- **PSIR2 (S8)** : expérimentation exploratoire en vision aérienne civile, à partir de recadrages VisDrone, avec ResNet-50 et DeiT-Tiny. Graphiques de précision, matrices de confusion, carte d'activation ResNet et étude ponctuelle de la confiance.
- **PARE1 (S9)** : actualisation bibliographique, réexamen critique des données et résultats et rédaction d'un article de **8 à 10 pages maximum** selon le modèle ESAIP. La formulation doit distinguer constats observables, hypothèses et extrapolations interdites.

## Audit initial

Voir [AUDIT_SCIENTIFIQUE.md](AUDIT_SCIENTIFIQUE.md). Les résultats historiques ne sont ni remplacés ni validés par ce dépôt. Aucun nouvel entraînement scientifique n'a été exécuté sur VisDrone.

## Outil d'analyse statistique

L'outil `analyse_predictions.py` **ne réalise pas d'entraînement, de génération d'attaques ou d'amélioration du ciblage**. Il sert à examiner des prédictions préexistantes sur des images civiles.

CSV minimal (un exemple entièrement fictif figure dans `exemple_predictions.csv`) :

```csv
source_id,sample_id,split,y_true,resnet_pred,deit_pred
image_01,crop_01,test,scene_A,scene_A,scene_A
image_02,crop_02,test,scene_B,scene_A,scene_B
```

`source_id` désigne l'image d'origine avant recadrage : tous ses recadrages doivent appartenir à la même partition. `sample_id` doit être unique. Les classes sont des étiquettes abstraites de scènes civiles. Les entrées n'étant pas des scores de confiance, cet outil ne calcule pas de calibration.

Commande :

```powershell
python PARE1/analyse_predictions.py --csv PARE1/exemple_predictions.csv --out PARE1/resultats_exemple.json
python -m unittest discover -s PARE1/tests -v
```

Sortie : accuracy, balanced accuracy, matrices de confusion, supports de classe et écart apparié avec intervalle descriptif par bootstrap des **images sources**. L'IC est absent lorsqu'il n'y a qu'une image source. Sur de petits effectifs, un IC bootstrap ne prouve pas la significativité ni la généralisabilité.

## Suite du PARE1

1. Archiver les versions historiques des scripts, graphiques, modèle et environnement matériel.
2. Compléter un journal d'expérience : taille des groupes, partitions par image source, prétraitement, nombre d'époques et version des bibliothèques.
3. Réconcilier les résultats PSIR2 et les revendications : noter celles qui sont non vérifiées.
4. Actualiser l'état de l'art avec les publications effectivement consultées.
5. Rédiger en suivant `TRAME_ARTICLE.md` puis faire valider par le tuteur **Ladji Adiaviakoye**.

Le benchmark EuroSAT réalisé séparément peut être mentionné comme **autre tâche exploratoire** ; ses résultats ne prouvent pas la généralisation de VisDrone.
