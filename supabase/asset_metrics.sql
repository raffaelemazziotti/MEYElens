-- Run once in the Supabase SQL Editor.
-- This is separate from the existing download_assets table.

create type public.asset_metrics_kind as enum ('ai_model', 'model_3d', 'tutorial');
create type public.asset_metrics_metric as enum ('download', 'view');

create table public.asset_metrics (
  asset_kind public.asset_metrics_kind not null,
  asset_id text not null check (asset_id ~ '^[a-z0-9-]+$'),
  metric public.asset_metrics_metric not null,
  title text not null,
  count bigint not null default 0 check (count >= 0),
  updated_at timestamptz not null default now(),
  primary key (asset_kind, asset_id, metric)
);

insert into public.asset_metrics (asset_kind, asset_id, metric, title) values
  ('ai_model', 'segformer-light-v2', 'download', 'SegFormer Light V2'),
  ('ai_model', 'segformer-medium-v2', 'download', 'SegFormer Medium V2'),
  ('ai_model', 'segformer-large-v2', 'download', 'SegFormer Large V2'),
  ('ai_model', 'segformer-light', 'download', 'SegFormer Light'),
  ('ai_model', 'segformer-medium', 'download', 'SegFormer Medium'),
  ('ai_model', 'efficientformerv2-light', 'download', 'EfficientFormerV2 Light'),
  ('ai_model', 'efficientformerv2-small', 'download', 'EfficientFormerV2 Small'),
  ('ai_model', 'efficientformerv2-medium', 'download', 'EfficientFormerV2 Medium'),
  ('ai_model', 'efficientformerv2-large', 'download', 'EfficientFormerV2 Large'),
  ('ai_model', 'meye-pupil', 'download', 'MEYE pupil segmentation'),
  ('model_3d', 'printable-stl-package', 'download', 'Printable STL package'),
  ('model_3d', 'front-panel', 'download', 'Front panel'),
  ('model_3d', 'arm-left', 'download', 'Left arm'),
  ('model_3d', 'camera-arm', 'download', 'Camera arm'),
  ('model_3d', 'arm-right', 'download', 'Right arm'),
  ('model_3d', 'camera-holder', 'download', 'Camera holder'),
  ('model_3d', 'nose-support', 'download', 'Nose support'),
  ('model_3d', 'adjustment-knob', 'download', 'Adjustment knob'),
  ('model_3d', 'ear-support-left', 'download', 'Left ear support'),
  ('model_3d', 'ear-support-right', 'download', 'Right ear support'),
  ('model_3d', 'external-ir-support', 'download', 'External IR support'),
  ('tutorial', 'feature-preview', 'view', 'Preview and feature extraction'),
  ('tutorial', 'change-model', 'view', 'Use a custom model'),
  ('tutorial', 'online-meye-recording', 'view', 'Online MEYElens recording'),
  ('tutorial', 'pupillary-light-reflex', 'view', 'Pupillary light reflex'),
  ('tutorial', 'visual-oddball', 'view', 'Visual oddball task'),
  ('tutorial', 'raw-eye-video-recording', 'view', 'Raw eye-video recording'),
  ('tutorial', 'eye-dataset-collector', 'view', 'Eye-image dataset collector'),
  ('tutorial', 'one-camera-gaze-calibration', 'view', 'One-camera gaze calibration');

alter table public.asset_metrics enable row level security;
grant select on public.asset_metrics to anon, authenticated;

create policy "Anyone can read asset metrics"
on public.asset_metrics
for select
to anon, authenticated
using (true);

create or replace function public.increment_asset_metric(
  p_asset_kind public.asset_metrics_kind,
  p_asset_id text,
  p_metric public.asset_metrics_metric
)
returns bigint
language plpgsql
security definer
set search_path = public, pg_catalog
as $$
declare
  new_count bigint;
begin
  update public.asset_metrics
  set count = count + 1, updated_at = now()
  where asset_kind = p_asset_kind
    and asset_id = p_asset_id
    and metric = p_metric
  returning count into new_count;

  if new_count is null then
    raise exception 'Unknown tracked asset or metric';
  end if;

  return new_count;
end;
$$;

revoke all on function public.increment_asset_metric(public.asset_metrics_kind, text, public.asset_metrics_metric) from public;
grant execute on function public.increment_asset_metric(public.asset_metrics_kind, text, public.asset_metrics_metric) to anon, authenticated;
