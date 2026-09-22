/**
 * ZeroDataModel TypeScript SDK.
 *
 * A thin asynchronous client for the ZeroDataModel REST API covering the core
 * cognitive endpoints and the phase-7 cognitive-upgrade modules.
 */
export { ZeroDataModelClient } from './client';
export {
  ZeroDataModelError,
  AuthenticationError,
  RateLimitError,
  NotFoundError,
  ValidationError,
  ServerError,
} from './errors';
export { ClientConfig } from './config';
export type { ClientConfigOptions } from './config';
export type {
  HealthResponse,
  ThinkResponse,
  KnowledgeGraphResponse,
  JsonObject,
  ArchitectActionRequest,
  LayeredPredictorUpdateRequest,
  EpisodicGraphPlanRequest,
  SemanticIndexSearchRequest,
  ExperimentResultRequest,
  HypothesisTestRequest,
} from './types';

export const VERSION = '1.0.0';
