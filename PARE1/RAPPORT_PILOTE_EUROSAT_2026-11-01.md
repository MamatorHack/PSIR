# PARE1 — Résultats contrôlés du pilote civil EuroSAT RGB (1er novembre 2026)

**Source analysée :** `PARE1_EuroSAT_pilote_resultats.zip` transmis par l'équipe. Résultats expérimentaux réels, mais **exploratoires**, distincts du PSIR2 historique sur VisDrone.

## Conditions observées

- **300 images par classe**, 10 classes, soit 3 000 images ; partitions train **2 100**, val **450**, test **450**.
- **Graine 42 uniquement** ; `split_hash = aa33a2a201781653d7b484139354ce99ab1aff4c647b412e4fbe2c171b791d51`.
- ResNet-50 (CNN) et DeiT-Tiny (Vision Transformer), poids préentraînés, **5 époques max**, LR 0,0001, batch 32.
- Entraînement sur **Tesla T4**, Python 3.13.16, PyTorch 2.11.0+cu130.
- Évaluation des mêmes 450 images en condition propre et **FGSM white-box non ciblé, L-infini, ε=2/255**, perturbations calculées séparément pour chaque modèle.
- Vérification : aucun chemin de fichier exactement dupliqué dans les 3 000 entrées du manifeste, partitions équilibrées à 45 images/classe pour le test ; mêmes identifiants et étiquettes entre les deux modèles. Cela **n'exclut pas** une corrélation géographique entre photos.

## Résultats

| Condition | ResNet-50 | DeiT-Tiny | Écart DeiT − ResNet (points) | IC bootstrap 95 % |
|---|---:|---:|---:|---|
| Propre | 92,00 % (414/450) | 94,22 % (424/450) | +2,22 | [−0,44 ; +4,67] |
| FGSM ε=2/255 | 55,56 % (250/450) | 26,89 % (121/450) | −28,67 | [−34,67 ; −22,22] |

Chute propre → FGSM : **−36,44 points** pour ResNet-50 et **−67,33 points** pour DeiT-Tiny.

**Interprétation :** sur les images propres, l'IC de l'écart inclut zéro ; la supériorité nominale de DeiT n'est pas établie. Sous **cette** condition FGSM, ResNet conserve une précision plus haute. Il ne faut pas extrapoler à toutes les architectures ou perturbations, ni à VisDrone.

## Entraînement

- ResNet-50 validation : **56,67 % à l'époque 1**, **92,67 % à l'époque 5** (meilleure valeur observée à la dernière époque) ; convergence non établie.
- DeiT-Tiny validation : **91,78 % à l'époque 1**, meilleur **96,00 % à l'époque 4**, **94,89 % à l'époque 5**.

## Confiance et calibration

| Condition | ResNet (confiance top-1 ; ECE) | DeiT (confiance top-1 ; ECE) |
|---|---|---|
| Propre | 85,29 % ; 8,07 % | 95,21 % ; 2,53 % |
| FGSM | 44,67 % ; 10,89 % | 84,84 % ; 58,52 % |

La confiance de DeiT reste élevée sous FGSM malgré une chute de précision. L'ECE est un estimateur dépendant du choix de discrétisation.

## Variabilité entre classes

Sous FGSM, **Forest** : ResNet 0/45, DeiT 32/45 ; **River** : ResNet 37/45, DeiT 1/45 ; **Residential** : ResNet 39/45, DeiT 12/45. Pas de conclusion générale par classe à ce stade.

## Latence GPU

Sur Tesla T4, **batch 1**, 20 mesures : médiane ResNet **6,25 ms**, DeiT **5,01 ms**. Ce n'est **pas** une mesure sur matériel embarqué.

## Portée scientifique

Ce pilote prouve que les deux pipelines apprennent et permet une analyse comparative exploratoire sur images satellitaires civiles. Il **ne démontre pas** une architecture hybride CNN–Transformer, une robustesse générale, un transfert de domaine, la fiabilité de drones ou la sûreté militaire. Les valeurs historiques du PSIR2 ne sont pas confirmées ici.

L'échantillon ne comprend qu'**une seule graine**, et les données satellitaires peuvent être spatialement corrélées ; les IC descriptifs peuvent surestimer la précision de l'inférence.

## Suite recommandée

Lancer une **campagne intermédiaire civile pré-définie**, avant une expérience plus ambitieuse :
`MAX_PER_CLASS=600`, `EPOCHS=8`, `SEEDS=[42,123,2026]`, `ATTACKS=['fgsm']`, `EPSILONS=[2/255]`.

Conserver pour chaque graine les manifestes de split, paramètres, historiques, métriques, prédictions et graphiques. Les splits **changent avec la graine**, donc leur dispersion ne mesure pas uniquement l'aléa d'initialisation. Présenter ces conclusions avec prudence et les faire valider par le tuteur.

**Rappel :** l'article officiel PARE1 doit suivre la trame Moodle, mettre à jour l'état de l'art et conserver explicitement la continuité PSIR1 → PSIR2 → PARE1.
