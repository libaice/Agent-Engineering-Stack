create table runs (
  run_id text primary key,
  thread_id text not null,
  user_id text not null,
  org_id text not null,
  question text not null,
  status text not null,
  answer text,
  answerable boolean,
  confidence double precision,
  metadata jsonb default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);