# PARE1 — Campagne intermédiaire EuroSAT RGB (01/11/2026)

**Provenance :** archive `PARE1_EuroSAT_intermediaire_resultats.zip` transmise par l'équipe le 1er novembre 2026. Il s'agit de résultats civils **distincts** du PSIR2 historique VisDrone ; les modèles et données historiques de PSIR2 ne sont pas reconstruits ici.

## Protocole constaté
- Classification EuroSAT RGB en 10 classes, 600 images/classe par exécution.
- Graines : 42, 123, 2026. Pour chacune : 4 200 images d'entraînement, 900 de validation et 900 de test (90/classe), 8 époques maximum, batch 32, LR=0,0001, poids préentraînés, Tesla T4.
- ResNet-50 (CNN) vs DeiT-Tiny (Vision Transformer), évalués sur les mêmes 900 images de test **à l'intérieur d'une graine**.
- Deux conditions : images propres et FGSM non ciblé **white-box**, norme L∞, ε=2/255, attaques calculées séparément à partir des gradients de chaque modèle.

## Résultats (accuracy)
| Graine | ResNet propre | DeiT propre | ResNet FGSM | DeiT FGSM | Écart FGSM DeiT − ResNet |
|---|---:|---:|---:|---:|---:|
| 42 | 97,00 % | 95,78 % | 58,89 % | 22,89 % | −36,00 pp |
| 123 | 96,78 % | 96,56 % | 58,56 % | 31,56 % | −27,00 pp |
| 2026 | 96,67 % | 96,22 % | 57,33 % | 27,44 % | −29,89 pp |
| **Moyenne (3 graines)** | **96,81 %** | **96,19 %** | **58,26 %** | **27,30 %** | **−30,96 pp** |

Dispersion entre graines (écart-type descriptif) : sans perturbation ResNet ±0,17 pp, DeiT ±0,39 pp ; FGSM ResNet ±0,82 pp, DeiT ±4,34 pp.

Les IC bootstrap appariés à 95 % par graine (écart DeiT − ResNet) :
- **Propre** : graine 42 [−2,67 ; +0,22] ; 123 [−1,67 ; +1,00] ; 2026 [−2,00 ; +1,00] pp : tous incluent zéro.
- **FGSM** : graine 42 [−39,78 ; −31,89] ; 123 [−30,67 ; −23,22] ; 2026 [−33,89 ; −25,89] pp.

## Autres métriques
Sous FGSM, confiance maximale Softmax moyenne des 3 exécutions : ResNet **71,42 %**, DeiT **86,87 %** ; ECE moyenne : ResNet **13,81 %**, DeiT **59,57 %**. Le haut niveau de confiance de DeiT malgré ses erreurs illustre une limite de la confiance top-1 comme indicateur de fiabilité.

Hétérogénéité des classes : sous FGSM, pour **Forest**, ResNet réussit [9,2,0] images sur 90 et DeiT [46,40,32] ; pour **Residential**, ResNet [69,82,75] et DeiT [30,64,29]. Le classement peut s'inverser selon la classe.

## Vérifications et réserves
- Les trois splits ont été contrôlés depuis `split_manifest.csv` : aucune **identité de chemin** entre train/validation/test dans une même graine.
- Les métriques de chaque modèle ont été recalculées depuis `test_predictions.csv` (900 observations/condition/modèle/graine) : elles concordent avec les JSON.
- **Pas 2 700 observations de test indépendantes** : les ensembles de test de 900 images se recoupent entre graines (42, 35 et 40 images en commun par paire ; union de 2 584 images). Les splits et sous-échantillons changent aussi avec la graine.
- Les chemins distincts n'excluent pas des scènes proches ou une corrélation spatiale non documentée.
- Seulement **une** intensité FGSM ; ne pas extrapoler à toutes les attaques, tous les modèles ou tous les domaines.
- EuroSAT : scènes satellitaires civiles ; **ce travail ne démontre aucune propriété de sûreté d'un drone militaire ni l'hypothèse d'une architecture hybride CNN–Transformer**.

## Conclusion scientifique proposée
Sur ce protocole civil, les deux architectures ont des précisions proches sur les images propres (environ 96–97 %). Le ResNet-50 conserve sur les trois exécutions une précision supérieure au DeiT-Tiny pour **l'unique perturbation FGSM expérimentée**. Cette observation contredit une hypothèse de supériorité systématique du Transformer, mais ne généralise pas à d'autres perturbations ni à la tâche VisDrone.

## Prochaine étape PARE1
Conserver les métriques et fichiers source originaux, intégrer cette étude complémentaire dans la **discussion méthodologique** du PARE1, actualiser l'état de l'art et soumettre l'orientation au tuteur. Le manuscrit doit explicitement distinguer PSIR1 (hypothèses), PSIR2 (exploration VisDrone) et PARE1 (audit et contre-exemple civil EuroSAT).
