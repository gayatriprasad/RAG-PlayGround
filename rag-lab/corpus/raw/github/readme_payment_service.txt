# README: Payment Service (payment-svc)

## Overview
The payment service handles all payment processing, subscription management, and billing operations for Acme Corp. It integrates with Stripe for payment processing and manages the subscription lifecycle.

## Architecture
- **Language**: Go 1.21
- **Framework**: Custom HTTP framework (pkg/httpkit)
- **Database**: PostgreSQL 15 (billing_db)
- **Message Queue**: Kafka (payment events)
- **External**: Stripe API (payments), TaxJar (tax calculation)

## Key Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v2/payments/charge | Process one-time payment |
| POST | /api/v2/subscriptions | Create subscription |
| PATCH | /api/v2/subscriptions/:id | Update subscription (upgrade/downgrade) |
| DELETE | /api/v2/subscriptions/:id | Cancel subscription |
| GET | /api/v2/invoices | List invoices for org |
| POST | /api/v2/webhooks/stripe | Stripe webhook receiver |

## Subscription Tiers
| Tier | Price | Features |
|------|-------|----------|
| Free | $0/mo | 5 users, 1GB storage, basic search |
| Pro | $29/user/mo | Unlimited users, 100GB storage, advanced search, API access |
| Enterprise | Custom | SSO, SLA, dedicated support, custom integrations |

## Local Development
```bash
# Prerequisites: Docker, Go 1.21+
docker compose up -d postgres redis stripe-mock
go run cmd/payment-svc/main.go

# Run tests
go test ./... -v -count=1

# Run with Stripe test keys
STRIPE_SECRET_KEY=sk_test_xxx go run cmd/payment-svc/main.go
```

## Deployment
- Deployed via ArgoCD to Kubernetes
- 3 replicas in production, HPA scales to 10
- Health check: GET /healthz
- Readiness: GET /readyz (checks DB + Stripe connectivity)

## Stripe Integration
- Webhook signature verification: REQUIRED (reject unsigned events)
- Idempotency keys: used for all write operations
- Retry policy: exponential backoff, max 5 retries
- Events processed: payment_intent.succeeded, invoice.paid, subscription.updated, subscription.deleted

## Monitoring
- Dashboard: "Payment Service Health" in Datadog
- Key metrics: payment success rate, p99 latency, webhook processing time
- Alerts: success rate <99.5%, p99 >2s, webhook backlog >100

## On-Call
- Team: Payments & Billing (Slack: #payments-team)
- Runbook: docs.internal.acme.com/runbooks/payment-svc
- Escalation: PagerDuty → EM → Director → VP Eng
