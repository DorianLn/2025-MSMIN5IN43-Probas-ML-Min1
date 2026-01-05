import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionInterval {
  prediction: number;
  lower: number;
  upper: number;
}

@Injectable({
  providedIn: 'root'
})
export class PredictionService {

  private apiUrl = 'http://localhost:8000'; // URL de l'API FastAPI

  constructor(private http: HttpClient) { }

  predict(features: any, alpha: number): Observable<PredictionInterval> {
    const payload = {
      alpha: alpha,
      features: features
    };
    return this.http.post<PredictionInterval>(`${this.apiUrl}/predict_single`, payload);
  }
}
