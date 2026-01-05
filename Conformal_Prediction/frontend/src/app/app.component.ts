import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { FormsModule } from '@angular/forms';

// Angular Material Modules
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatExpansionModule } from '@angular/material/expansion';


import { AmesData } from './ames-data.model';
import { PredictionService, PredictionInterval } from './prediction.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterOutlet,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSliderModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatExpansionModule
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  confidence: number = 95;
  lowerBound: number | null = null;
  upperBound: number | null = null;
  prediction: number | null = null;
  isFading: boolean = false;
  isLoading: boolean = false;
  predictionError: string | null = null;

  // Initialiser avec des valeurs par défaut raisonnables ou vides
  formData: AmesData = {
    'MS SubClass': 20,
    'MS Zoning': 'RL',
    'Lot Frontage': 80,
    'Lot Area': 9600,
    'Street': 'Pave',
    'Alley': null,
    'Lot Shape': 'Reg',
    'Land Contour': 'Lvl',
    'Utilities': 'AllPub',
    'Lot Config': 'Inside',
    'Land Slope': 'Gtl',
    'Neighborhood': 'NAmes',
    'Condition 1': 'Norm',
    'Condition 2': 'Norm',
    'Bldg Type': '1Fam',
    'House Style': '1Story',
    'Overall Qual': 6,
    'Overall Cond': 5,
    'Year Built': 1970,
    'Year Remod/Add': 1970,
    'Roof Style': 'Gable',
    'Roof Matl': 'CompShg',
    'Exterior 1st': 'HdBoard',
    'Exterior 2nd': 'HdBoard',
    'Mas Vnr Type': 'None',
    'Mas Vnr Area': 0,
    'Exter Qual': 'TA',
    'Exter Cond': 'TA',
    'Foundation': 'CBlock',
    'Bsmt Qual': 'TA',
    'Bsmt Cond': 'TA',
    'Bsmt Exposure': 'No',
    'BsmtFin Type 1': 'ALQ',
    'BsmtFin SF 1': 978,
    'BsmtFin Type 2': 'Unf',
    'BsmtFin SF 2': 0,
    'Bsmt Unf SF': 284,
    'Total Bsmt SF': 1262,
    'Heating': 'GasA',
    'Heating QC': 'Ex',
    'Central Air': 'Y',
    'Electrical': 'SBrkr',
    '1st Flr SF': 1262,
    '2nd Flr SF': 0,
    'Low Qual Fin SF': 0,
    'Gr Liv Area': 1262,
    'Bsmt Full Bath': 0,
    'Bsmt Half Bath': 1,
    'Full Bath': 2,
    'Half Bath': 0,
    'Bedroom AbvGr': 3,
    'Kitchen AbvGr': 1,
    'Kitchen Qual': 'TA',
    'TotRms AbvGrd': 6,
    'Functional': 'Typ',
    'Fireplaces': 1,
    'Fireplace Qu': 'TA',
    'Garage Type': 'Attchd',
    'Garage Yr Blt': 1976,
    'Garage Finish': 'RFn',
    'Garage Cars': 2,
    'Garage Area': 460,
    'Garage Qual': 'TA',
    'Garage Cond': 'TA',
    'Paved Drive': 'Y',
    'Wood Deck SF': 298,
    'Open Porch SF': 0,
    'Enclosed Porch': 0,
    '3Ssn Porch': 0,
    'Screen Porch': 0,
    'Pool Area': 0,
    'Pool QC': null,
    'Fence': null,
    'Misc Feature': null,
    'Misc Val': 0,
    'Mo Sold': 5,
    'Yr Sold': 2007,
    'Sale Type': 'WD',
    'Sale Condition': 'Normal'
  };

  featureGroups = [
    {
      name: 'Général',
      features: [
        { original: 'MS SubClass', display: 'Sous-classe de la propriété' },
        { original: 'MS Zoning', display: 'Classement du zonage' },
        { original: 'Neighborhood', display: 'Quartier' },
        { original: 'Bldg Type', display: 'Type de bâtiment' },
        { original: 'House Style', display: 'Style de la maison' }
      ]
    },
    {
      name: 'Lot',
      features: [
        { original: 'Lot Frontage', display: 'Façade du terrain' },
        { original: 'Lot Area', display: 'Superficie du terrain' },
        { original: 'Street', display: 'Type d\'accès à la rue' },
        { original: 'Alley', display: 'Accès à l\'allée' },
        { original: 'Lot Shape', display: 'Forme du terrain' },
        { original: 'Land Contour', display: 'Topographie du terrain' },
        { original: 'Utilities', display: 'Services publics' },
        { original: 'Lot Config', display: 'Configuration du lot' },
        { original: 'Land Slope', display: 'Pente du terrain' }
      ]
    },
    {
      name: 'Qualité & Condition',
      features: [
        { original: 'Overall Qual', display: 'Qualité générale du matériau et de la finition' },
        { original: 'Overall Cond', display: 'État général' },
        { original: 'Exter Qual', display: 'Qualité de l\'extérieur' },
        { original: 'Exter Cond', display: 'État de l\'extérieur' },
        { original: 'Bsmt Qual', display: 'Hauteur du sous-sol' },
        { original: 'Bsmt Cond', display: 'État général du sous-sol' },
        { original: 'Heating QC', display: 'Qualité et état du chauffage' },
        { original: 'Kitchen Qual', display: 'Qualité de la cuisine' },
        { original: 'Fireplace Qu', display: 'Qualité du foyer' },
        { original: 'Garage Qual', display: 'Qualité du garage' },
        { original: 'Garage Cond', display: 'État du garage' },
        { original: 'Pool QC', display: 'Qualité de la piscine' },
        { original: 'Fence', display: 'Qualité de la clôture' }
      ]
    },
    {
      name: 'Construction',
      features: [
        { original: 'Year Built', display: 'Date de construction originale' },
        { original: 'Year Remod/Add', display: 'Date de rénovation/agrandissement' },
        { original: 'Roof Style', display: 'Type de toit' },
        { original: 'Roof Matl', display: 'Matériau du toit' },
        { original: 'Exterior 1st', display: 'Revêtement extérieur 1' },
        { original: 'Exterior 2nd', display: 'Revêtement extérieur 2' },
        { original: 'Mas Vnr Type', display: 'Type de placage de maçonnerie' },
        { original: 'Mas Vnr Area', display: 'Superficie du placage de maçonnerie en pieds carrés' },
        { original: 'Foundation', display: 'Type de fondation' },
        { original: 'Heating', display: 'Type de chauffage' },
        { original: 'Central Air', display: 'Climatisation centrale' },
        { original: 'Electrical', display: 'Système électrique' },
        { original: 'Functional', display: 'Évaluation fonctionnelle de la maison' }
      ]
    },
    {
      name: 'Sous-sol',
      features: [
        { original: 'Bsmt Exposure', display: 'Exposition du sous-sol' },
        { original: 'BsmtFin Type 1', display: 'Qualité de finition du sous-sol (Type 1)' },
        { original: 'BsmtFin SF 1', display: 'Superficie finie du sous-sol (Type 1) en pieds carrés' },
        { original: 'BsmtFin Type 2', display: 'Qualité de finition du sous-sol (Type 2)' },
        { original: 'BsmtFin SF 2', display: 'Superficie finie du sous-sol (Type 2) en pieds carrés' },
        { original: 'Bsmt Unf SF', display: 'Superficie non finie du sous-sol en pieds carrés' },
        { original: 'Total Bsmt SF', display: 'Superficie totale du sous-sol en pieds carrés' },
        { original: 'Bsmt Full Bath', display: 'Salles de bain complètes au sous-sol' },
        { original: 'Bsmt Half Bath', display: 'Demi-salles de bain au sous-sol' }
      ]
    },
    {
      name: 'Espaces de vie',
      features: [
        { original: '1st Flr SF', display: 'Superficie du premier étage en pieds carrés' },
        { original: '2nd Flr SF', display: 'Superficie du deuxième étage en pieds carrés' },
        { original: 'Low Qual Fin SF', display: 'Superficie finie de faible qualité en pieds carrés' },
        { original: 'Gr Liv Area', display: 'Superficie habitable au-dessus du sol en pieds carrés' },
        { original: 'Full Bath', display: 'Salles de bain complètes au-dessus du sol' },
        { original: 'Half Bath', display: 'Demi-salles de bain au-dessus du sol' },
        { original: 'Bedroom AbvGr', display: 'Chambres au-dessus du sol' },
        { original: 'Kitchen AbvGr', display: 'Cuisines au-dessus du sol' },
        { original: 'TotRms AbvGrd', display: 'Pièces totales au-dessus du sol (sans salles de bain)' },
        { original: 'Fireplaces', display: 'Nombre de foyers' }
      ]
    },
    {
      name: 'Garage',
      features: [
        { original: 'Garage Type', display: 'Localisation du garage' },
        { original: 'Garage Yr Blt', display: 'Année de construction du garage' },
        { original: 'Garage Finish', display: 'Finition intérieure du garage' },
        { original: 'Garage Cars', display: 'Taille du garage en capacité de voiture' },
        { original: 'Garage Area', display: 'Superficie du garage en pieds carrés' },
        { original: 'Paved Drive', display: 'Allée pavée' }
      ]
    },
    {
      name: 'Extérieur',
      features: [
        { original: 'Wood Deck SF', display: 'Superficie du pont en bois en pieds carrés' },
        { original: 'Open Porch SF', display: 'Superficie du porche ouvert en pieds carrés' },
        { original: 'Enclosed Porch', display: 'Superficie du porche fermé en pieds carrés' },
        { original: '3Ssn Porch', display: 'Superficie du porche trois saisons en pieds carrés' },
        { original: 'Screen Porch', display: 'Superficie du porche grillagé en pieds carrés' },
        { original: 'Pool Area', display: 'Superficie de la piscine en pieds carrés' },
        { original: 'Misc Feature', display: 'Caractéristique diverse non couverte par d\'autres catégories' },
        { original: 'Misc Val', display: 'Valeur de la caractéristique diverse' }
      ]
    },
    {
      name: 'Vente',
      features: [
        { original: 'Mo Sold', display: 'Mois de vente' },
        { original: 'Yr Sold', display: 'Année de vente' },
        { original: 'Sale Type', display: 'Type de vente' },
        { original: 'Sale Condition', display: 'Condition de vente' }
      ]
    }
  ];

  numericFields: (keyof AmesData)[] = [
    'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
    'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
    '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',
    'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr',
    'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Garage Area',
    'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
    'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'
  ];

  constructor(private predictionService: PredictionService) {}

  isNumeric(field: keyof AmesData): boolean {
    return this.numericFields.includes(field);
  }

    onSubmit(): void {
      this.isLoading = true;
      this.isFading = true;
      this.predictionError = null;
      this.lowerBound = null;
      this.upperBound = null;
      this.prediction = null;
  
      // Convertir les valeurs vides en null
      const features: any = { ...this.formData };
      for (const key in features) {
        if (features[key] === '' || features[key] === undefined) {
          features[key] = null;
        }
      }
      
      // L'alpha est 1 - (niveau de confiance / 100)
      const alpha = 1 - (this.confidence / 100);
  
      this.predictionService.predict(features, alpha).subscribe({
        next: (result: any) => {
          this.lowerBound = result.lower;
          this.upperBound = result.upper;
          this.prediction = result.prediction;
          this.isLoading = false;
          setTimeout(() => { this.isFading = false; }, 50);
        },
        error: (err) => {
          console.error('Erreur lors de la prédiction:', err);
          this.predictionError = 'Une erreur est survenue lors de la communication avec le serveur. Vérifiez la console pour plus de détails.';
          this.isLoading = false;
          this.isFading = false;
        }
      });
    }}
