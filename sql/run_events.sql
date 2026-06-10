create table run_events (
  id bigserial primary key,
  run_id text not null references runs(run_id),
  thread_id text not null,
  event_type text not null,
  node text,
  payload jsonb not null,
  created_at timestamptz not null default now()
);