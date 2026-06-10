create table memories (
  memory_id text primary key,
  user_id text not null,
  org_id text not null,
  type text not null,
  name text not null,
  content text not null,
  aliases jsonb default '[]',
  importance double precision default 0.5,
  metadata jsonb default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);