create table interrupts (
  interrupt_id text primary key,
  run_id text not null references runs(run_id),
  thread_id text not null,
  type text not null,
  payload jsonb not null,
  status text not null,
  resume_value jsonb,
  created_at timestamptz not null default now(),
  resolved_at timestamptz
);