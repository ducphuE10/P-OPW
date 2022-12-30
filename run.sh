for dataset in Fungi GunPoint Plane MoteStrain FaceFour OliveOil DistalPhalanxTW ToeSegmentation2 DistalPhalanxOutlineCorrect Herring BME ECG200 BeetleFly BirdChicken;
do for lambda1 in 10
do
    python test.py --lambda1=$lambda1 --lambda2=0.1 --dataset=$dataset --m=0.8 --normalize
done
done