# Audit scientifique — PSIR2 vers PARE1

**Base auditée :** scripts historiques à la racine du dépôt, rapport PSIR2 et documents PSIR1 fournis par l'équipe. Ce document distingue ce qui est visible dans le code de ce qui exigerait les données, checkpoints et journaux originaux.

| Élément du PSIR2 | État de vérification et incidence scientifique |
|---|---|
| Jeu de données | `00_prepare_proxy_dataset.py` télécharge le *set de validation* VisDrone2019-DET-val et recadre jusqu'à 200 exemples par classe. Ce jeu sert ensuite à l'entraînement et à la validation : il ne reste donc pas un jeu de validation officiel indépendant. |
| Divisions | `01_train.py` partitionne aléatoirement les **recadrages** à 80/20. Plusieurs objets d'une même photographie peuvent être présents des deux côtés, créant un risque de fuite de contexte. Il faudrait évaluer les partitions par photo d'origine avant toute interprétation confirmatoire. |
| Entraînement | `EPOCHS=2`, `PATIENCE=1`. Cela correspond à une preuve de concept, pas à un entraînement convergé. La description de « transfert partiel / couches gelées » n'est pas conforme au code : aucune couche n'est explicitement gelée. |
| Mesure des performances | Le temps par image est chronométré sur la machine qui exécute le code. Il ne peut attester un délai garanti sur FPGA, Jetson ou drone non testés. Matériel et tailles de batch doivent être mentionnés. |
| Test statistique | `02_evaluate.py` réutilise le découpage 80/20, limité à **20 exemples**. Une précision de 5 % signifie une prédiction correcte sur 20 ; variation de 5 points par image. Des résultats à 20 images sont illustratifs, non conclusifs. |
| Protocole adversarial | `02_evaluate.py` génère séparément FGSM et PGD avec les gradients de chaque modèle : il s'agit de **white-box**. Les termes « black-box » et « limite absolue » doivent être retirés des conclusions. |
| Modèles | ResNet-50 est un CNN ; DeiT-Tiny est un Vision Transformer. Ce n'est **pas** une démonstration expérimentale des propriétés d'un hybride CNN+ViT. |
| Cross-domain | `03_cross_domain_eval.py` charge le proxy source quand Target_Domain est absent. Dans cette configuration, les courbes ne démontrent aucun transfert vers un second domaine indépendant. |
| XAI | `04_explainability_confidence.py` génère une carte Grad-CAM du **ResNet seulement**, sur un exemple. Il ne démontre aucune carte équivalente de DeiT, ni mécanisme intrinsèque de verrouillage. |
| Softmax | La probabilité **maximale** d'un softmax de trois classes est toujours >= 1/3. Des interprétations rapportant <30 % sont incompatibles avec la métrique codée, hors autre mesure explicitée. Cette grandeur n'est pas une calibration ; il faut des étiquettes, des scores et des évaluations agrégées. |
| Figure et rapport | Les nombres 50 %, 5 %, 11,0 ms et 50,2 ms apparaissent dans le rendu historique. Sans checkpoints, logs et environnement matériel originaux, on ne peut pas certifier leur reproductibilité à partir du dépôt. |

## Position scientifique à défendre dans le PARE1

La contribution de PSIR2 est **une comparaison exploratoire entre deux familles de classificateurs sur des recadrages civils d'images aériennes**. Elle ne permet pas d'affirmer une supériorité systématique d'architecture, un transfert militaire, une immunité aux attaques, ni la sûreté d'une boucle décisionnelle autonome.

L'hypothèse d'hybridation formulée en PSIR1 peut rester dans **la discussion et les travaux futurs**, mais ne doit pas être décrite comme validée en PSIR2.

## Contrôles de qualité d'un futur jeu de résultats

- Enregistrer les identifiants de photographies sources pour repérer les recadrages corrélés.
- Séparer les partitions par image source (et, si possible, par séquence/lieu de prise de vue).
- Conserver les prédictions individuelles des deux architectures sur **les mêmes observations de test**.
- Indiquer les métriques, dénominateurs, graines et incertitudes ; rapporter aussi les expériences négatives.
- Fournir les versions exactes, le matériel et la provenance des fichiers.
- Ne pas comparer directement les chiffres EuroSAT et VisDrone comme s'il s'agissait du même problème.

## Remise en question scientifique

Cette correction méthodologique fait **partie du résultat PARE1** : elle donne au lecteur la possibilité de distinguer observations historiques et conclusion étayée. Aucune nouvelle valeur de performance n'est inventée ici.
